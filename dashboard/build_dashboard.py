"""Build dashboard/index.html: a single self-contained page (data inlined) from lewm_sweep/results/.
Usage: python dashboard/build_dashboard.py"""
import csv, json
from pathlib import Path
here = Path(__file__).resolve().parent
res = here.parent / "lewm_sweep" / "results"
data = {
    "pusht": list(csv.DictReader(open(res / "pusht_summary.csv"))),
    "cube": list(csv.DictReader(open(res / "cube_summary.csv"))),
    "difficulty": list(csv.DictReader(open(res / "pusht_episode_difficulty.csv"))),
}
html = (here / "template.html").read_text().replace("/*DATA*/null", json.dumps(data))
(here / "index.html").write_text(html)
print("wrote", here / "index.html", f"({len(html)} bytes)")
