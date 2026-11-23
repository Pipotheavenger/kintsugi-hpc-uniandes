"""
bridge_report_card.py — Kintsugi's native report card (kirad.utils.get_wandb_report_card_tsvs) for BRIDGE runs.

get_wandb_report_card_tsvs reads the task list from run.config["model_params"]["tasks"], but in this repo's
configs (and in the W&B run config) the tasks live under training_params.tasks, so it fails with KeyError 'tasks'.
It also reads data_params["<split>_split_or_dataset_version"] and data_params["data_sources"], an older config
layout; ours keeps the CSV of each split under data_params["metadata=..."][<split>]["df_or_dataset_version"].
This wrapper does NOT modify kirad: it only adds those keys to the run config object returned by wandb.Api().run
(in memory, nothing is written back to the run config) and then calls the native function unchanged.
Note: the native function also logs a `report-cards-<run_id>` artifact to the W&B run (wandb.init resume).

Usage (repo root, conda env kintsugi, on Hypatia):
    KINTSUGI_RAD_PATH=$PWD PYTHONPATH=$PWD python hpc/bridge_report_card.py <run_id> [...] --out ~/Kintusgi/results
Writes <out>/<run_id>_report_card_<task>.tsv
"""
import argparse
import os

import wandb

import kirad.utils as ku

ENTITY, PROJECT = "sebastian-valencia-world-bank-", "kintsugi-hpc-uniandes"


class _Api(wandb.Api):
    def run(self, path):
        r = super().run(path)
        mp = r.config.setdefault("model_params", {})
        if "tasks" not in mp:
            mp["tasks"] = r.config["training_params"]["tasks"]
        dp = r.config["data_params"]
        meta = next(v for k, v in dp.items() if k.startswith("metadata"))
        for split in ("train", "val", "test"):
            dp.setdefault(f"{split}_split_or_dataset_version",
                          "/".join(meta[split]["df_or_dataset_version"].split("/")[-2:]))
        dp.setdefault("data_sources", ["BRIDGE-MH baseline"])
        return r


ku.wandb.Api = _Api

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default=os.path.expanduser("~/Kintusgi/results"))
ap.add_argument("--objective", default="macro_f1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for rid in a.runs:
    tsvs = ku.get_wandb_report_card_tsvs(f"{ENTITY}/{PROJECT}/{rid}", threshold_objective=a.objective)
    for task, tsv in tsvs.items():
        with open(os.path.join(a.out, f"{rid}_report_card_{task}.tsv"), "w") as f:
            f.write(tsv)
        print("####", rid, task)
        print(tsv)
