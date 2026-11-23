"""
bridge_eval.py — confusion matrices and raw scores for DAM 1 runs (run on Hypatia, where `wandb login` is set).

For each run it downloads from W&B the score tables the model logs (`scores-<task>-<split>-<run_id>`,
written by OrdinalRegressionPLModule.on_validation_test_epoch_end in kirad/base_models.py):
the validation table of the best epoch (max metrics/val/sn_eq_sp@0%i) and the test table. Each row is one
recording: uuid, the model's score and the true PHQ/GAD.

What the score is (CORAL, kirad/losses.py): each task head outputs ONE logit x per 30 s window; the score is
the mean of x over the recording's windows. CORAL learns one bias b_k per possible cutoff k, and
P(score > k) = sigmoid(x + b_k). So the score is a latent severity (higher = more severe) and the probability of
being above a cutoff is obtained with that cutoff's bias, read here from the best checkpoint
(chkpts/<experiment_name>/epoch-*.ckpt, keys tasks.<task>.loss.biases). Thresholds and AUROC are the same with
the score or the probability (monotone transform).

Then, per task and cutoff (PHQ > 9, > 14; GAD > 4, > 9, > 14):
  - threshold = the score where sensitivity = specificity on VALIDATION (as Kintsugi does);
  - that threshold is applied to TEST -> TP, FP, FN, TN, sensitivity (= recall), specificity,
    precision (PPV), NPV, F1, accuracy.
Computed per recording (as the pipeline) and per person (mean score of the person's recordings).

Usage (repo root, conda env kintsugi):
    python hpc/bridge_eval.py <run_id> [<run_id> ...] --out ~/Kintusgi/results
Writes <out>/<run_id>_scores.csv (val + test, one row per recording and task) and <out>/<run_id>_confusion.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
import glob

import torch
import wandb

ENTITY, PROJECT = "sebastian-valencia-world-bank-", "kintsugi-hpc-uniandes"
MONITOR = "metrics/val/sn_eq_sp@0%i"
CUTOFFS = {"depression": (9, 14), "anxiety": (4, 9, 14)}


def table(api, task, split, rid, version):
    art = api.artifact(f"{ENTITY}/{PROJECT}/scores-{task}-{split}-{rid}:{version}")
    df = art.get("scores").get_dataframe()
    return df[["uuid", "scores", "labels"]].rename(columns={"scores": "score", "labels": "label"})


def coral_biases(experiment_name, chkpt_root):
    paths = sorted(glob.glob(os.path.join(chkpt_root, experiment_name, "epoch-*.ckpt")))
    if not paths:
        print(f"   (no checkpoint in {chkpt_root}/{experiment_name}: probabilities not computed)")
        return None, None
    sd = torch.load(paths[-1], map_location="cpu", weights_only=False)["state_dict"]
    keys = {t: [k for k in sd if f"tasks.{t}." in k and k.endswith("loss.biases")] for t in CUTOFFS}
    return {t: sd[ks[0]].float().numpy() for t, ks in keys.items() if ks}, paths[-1]


def best_val_version(api, rid):
    run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    vals = pd.Series([r[MONITOR] for r in run.scan_history(keys=[MONITOR]) if r.get(MONITOR) is not None], dtype=float)
    best = int(vals.values.argmax())  # 0-based index of the validation = version of the val score artifact
    n_art = sum(1 for art in run.logged_artifacts() if art.name.startswith(f"scores-depression-val-{rid}:"))
    assert n_art == len(vals), f"{n_art} val score artifacts but {len(vals)} validations"
    return f"v{best}", best + 1, len(vals)


def sn_eq_sp_threshold(score, y):
    best, thr = 9.0, None
    for t in np.unique(score):
        pred = score >= t
        sn = (pred & y).sum() / max(y.sum(), 1)
        sp = (~pred & ~y).sum() / max((~y).sum(), 1)
        if abs(sn - sp) < best:
            best, thr = abs(sn - sp), t
    return thr


def confusion(score, y, thr):
    pred = score >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    sn = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sn / (ppv + sn) if (ppv + sn) else np.nan
    return dict(n=len(y), positives=int(y.sum()), TP=tp, FP=fp, FN=fn, TN=tn, sensitivity_recall=sn, specificity=sp,
                precision_ppv=ppv, npv=npv, f1=f1, accuracy=(tp + tn) / len(y))


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default=os.path.expanduser("~/Kintusgi/results"))
ap.add_argument("--chkpts", default=os.path.expanduser("~/Kintusgi/kintsugi-hpc-uniandes/chkpts"))
a = ap.parse_args()
out = os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
api = wandb.Api()

for rid in a.runs:
    vver, vnum, nval = best_val_version(api, rid)
    name = api.run(f"{ENTITY}/{PROJECT}/{rid}").name
    biases, ckpt = coral_biases(name, os.path.expanduser(a.chkpts))
    print(f"== {rid} ({name}): best validation {vnum} of {nval} (val score table {vver}); checkpoint {ckpt}")
    scores, rows = [], []
    for task, cuts in CUTOFFS.items():
        val, test = table(api, task, "val", rid, vver), table(api, task, "test", rid, "latest")
        for d, split in ((val, "val"), (test, "test")):
            d = d.assign(task=task, split=split, user=d.uuid.str.split("__").str[0])
            if biases is not None and task in biases:
                for c in cuts:
                    d[f"prob_gt{c}"] = 1 / (1 + np.exp(-(d.score + biases[task][c])))
            scores.append(d)
        per_person = {s: d.assign(user=d.uuid.str.split("__").str[0]).groupby("user").agg(score=("score", "mean"), label=("label", "first"))
                      for s, d in (("val", val), ("test", test))}
        for c in cuts:
            for level, (v, t) in (("recording", (val, test)), ("person", (per_person["val"], per_person["test"]))):
                thr = sn_eq_sp_threshold(v.score.to_numpy(), (v.label > c).to_numpy())
                m = confusion(t.score.to_numpy(), (t.label > c).to_numpy(), thr)
                rows.append(dict(run=rid, task=task, cutoff=f">{c}", level=level, threshold=thr, **m))
    allsc = pd.concat(scores, ignore_index=True)
    cols = ["uuid", "user", "split", "task", "score", "label"] + sorted(c for c in allsc.columns if c.startswith("prob_"))
    allsc[cols].to_csv(
        os.path.join(out, f"{rid}_scores.csv"), index=False)
    cm = pd.DataFrame(rows)
    cm.to_csv(os.path.join(out, f"{rid}_confusion.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(cm[cm.level == "recording"].drop(columns=["run", "level"]).to_string(index=False))
