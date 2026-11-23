"""
bridge_subset.py — keep only some questionnaire versions from the DAM 1 CSVs (runs anywhere: only reads/writes CSVs).

Example (on Hypatia):
    python hpc/bridge_subset.py --src ~/Kintusgi/Data/bridge_baseline/dam1 \
        --out ~/Kintusgi/Data/bridge_baseline/dam1_v6 --versions v6
The audio is not copied: the configs keep audio_root pointing to the original dam1/audio folder.
The split is inherited from bridge_prepare.py (by person, stratified by version and PHQ > 9), so the
v6 subset of each split is itself a 70/15/15 split of the v6 people, stratified by PHQ > 9.
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--versions", nargs="+", required=True)
a = ap.parse_args()
src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
for s in ("train", "val", "test"):
    d = pd.read_csv(os.path.join(src, f"{s}.csv"))
    v = d[d.version.isin(a.versions)]
    v.to_csv(os.path.join(out, f"{s}.csv"), index=False)
    p = v.drop_duplicates("user")
    print(f"{s}: {p.user.nunique()} people, {len(v)} recordings, PHQ>9 {100 * (p.phq > 9).mean():.1f} %, "
          f"GAD>9 {100 * (p.gad > 9).mean():.1f} %, GAD>14 {(p.gad > 14).sum()} people")
