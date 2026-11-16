"""
wandb_history.py — download the logged history of W&B runs to CSV (run on Hypatia, where `wandb login` is set).

Usage (from the repo root):
    python hpc/wandb_history.py pnybari4 bnkqcsas --out ~/Kintusgi/results
Writes <out>/<run_id>_val.csv (one row per validation: step, epoch, every metrics/val/* column) and
<out>/<run_id>_train.csv (step, epoch, train loss), and prints a short summary per run.
"""
import argparse
import os

import pandas as pd
import wandb

PATH = "/sebastian-valencia-world-bank-/kintsugi-hpc-uniandes/runs/{}"
MONITOR = "metrics/val/sn_eq_sp@0%i"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default=os.path.expanduser("~/Kintusgi/results"))
a = ap.parse_args()
os.makedirs(os.path.expanduser(a.out), exist_ok=True)

api = wandb.Api()
for rid in a.runs:
    run = api.run(PATH.format(rid))
    h = pd.DataFrame(run.scan_history())
    step = "trainer/global_step" if "trainer/global_step" in h else "_step"
    keep = [c for c in [step, "epoch"] if c in h]
    val_cols = sorted(c for c in h.columns if "/val/" in c and not c.startswith("scores"))
    val = h[keep + val_cols].dropna(subset=[MONITOR]).reset_index(drop=True)
    tr_cols = [c for c in h.columns if "/train/" in c and "loss" in c]
    train = h[keep + tr_cols].dropna(subset=tr_cols, how="all").reset_index(drop=True)
    out = os.path.expanduser(a.out)
    val.to_csv(os.path.join(out, f"{rid}_val.csv"), index=False)
    train.to_csv(os.path.join(out, f"{rid}_train.csv"), index=False)
    b = val[MONITOR].idxmax()
    print(f"== {rid} ({run.name}, {run.state}): {len(val)} validations, {len(train)} train rows")
    print(f"   best {MONITOR} = {val[MONITOR].max():.4f} at validation {b + 1} "
          f"(epoch {val.loc[b].get('epoch')}, step {val.loc[b, step]})")
    for c in ["metrics/val/total_loss", "metrics/val/auroc@0%i", "depression/metrics/val/auroc@0%i",
              "anxiety/metrics/val/auroc@0%i"]:
        if c in val:
            print(f"   {c}: first {val[c].iloc[0]:.4f}  best-ckpt {val.loc[b, c]:.4f}  last {val[c].iloc[-1]:.4f}")
