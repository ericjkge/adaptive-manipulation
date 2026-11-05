import os

import sys, json
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

TIMING = {"solve_seconds": 0.0, "solve_calls": 0, "env_plans": 0, "call_seconds": [], "call_envs": [], "throttle_sleep": 0.0}

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to(cfg.solver.device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        # thermal throttle: after each cost evaluation (one CEM round), idle SWEEP_THROTTLE x its duration
        _throttle = float(os.environ.get("SWEEP_THROTTLE", "0"))
        if _throttle > 0:
            _orig_get_cost = model.get_cost
            def _throttled_get_cost(*a, **kw):
                t = time.time()
                c = _orig_get_cost(*a, **kw)
                if cfg.solver.device == "mps": torch.mps.synchronize()
                s0 = time.time()
                time.sleep(_throttle * (s0 - t))
                TIMING["throttle_sleep"] += time.time() - s0  # measured idle, subtracted from timing
                return c
            model.get_cost = _throttled_get_cost
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        _orig_solve = solver.solve
        def _timed_solve(info_dict, *a, **kw):
            n = len(next(iter(info_dict.values())))
            if cfg.solver.device == "mps": torch.mps.synchronize()
            t0 = time.time()
            slept0 = TIMING["throttle_sleep"]
            out = _orig_solve(info_dict, *a, **kw)
            if cfg.solver.device == "mps": torch.mps.synchronize()
            dt = time.time() - t0 - (TIMING["throttle_sleep"] - slept0)  # exclude throttle idle
            TIMING["solve_calls"] += 1
            TIMING["env_plans"] += n
            TIMING["call_seconds"].append(dt)
            TIMING["call_envs"].append(n)
            return out
        solver.solve = _timed_solve
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    # -- sweep record: one JSON per config, with per-episode successes
    rec = {
        "iterations": cfg.solver.n_steps,
        "samples": cfg.solver.num_samples,
        "topk": cfg.solver.topk,
        "receding_horizon": cfg.plan_config.receding_horizon,
        "horizon": cfg.plan_config.horizon,
        "num_eval": cfg.eval.num_eval,
        "seed": cfg.seed,
        "throttle": float(os.environ.get("SWEEP_THROTTLE", "0")),  # >0 means timings are inflated; use the timing benchmark
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        "wall_seconds": end_time - start_time,
        **TIMING,
        # first call includes GPU warmup, so exclude it from the per-plan time
        "seconds_per_plan": sum(TIMING["call_seconds"][1:]) / max(sum(TIMING["call_envs"][1:]), 1),
        "seconds_per_plan_incl_warmup": TIMING["solve_seconds"] / max(TIMING["env_plans"], 1),
    }
    out_json = cfg.get("sweep_out", None)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rec, indent=2))
        print("wrote", out_json)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
