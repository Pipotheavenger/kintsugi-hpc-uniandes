"""
bridge_compare.py — compare DAM 1 runs on the same footing (run on the Mac: it reads the clean survey base).

Inputs: <results>/<run>_scores.csv (hpc/bridge_eval.py) for each run, the clean base and data/clean/dam1/test.csv
(to know each test person's questionnaire version).
Outputs in <results>:
  compare_auroc.csv  test AUROC per run, task and cutoff, at recording and person level, on two test sets:
                     "own"  = the run's own test set (all versions: 664 people; v6 runs: 273 people)
                     "v6"   = only the 273 v6 test people (the same people for every run; the v6 split is a subset
                              of the all-versions split), so all runs are comparable
                     with a 95 % bootstrap interval resampling people (person level only; 1000 draws, seed 0).
  compare_paired.csv AUROC difference of each run minus the reference run on the v6 test people, paired
                     bootstrap over people (same draws for both runs).
  compare_corr.csv   person-level Spearman correlations on each run's own test set: model score vs true PHQ/GAD,
                     vs age and vs years of education (and, as reference, true PHQ/GAD vs age and education).
Usage:
  python hpc/bridge_compare.py --base Bridge_baseline --results Bridge_baseline/data/clean/results \
      --runs A=pnybari4 B=bnkqcsas C=qviai5hj D=oxmqyl0h E=jmy587yu --ref A
"""
import argparse
import os

import numpy as np
import pandas as pd

CUTOFFS = {"depression": (9, 14), "anxiety": (4, 9, 14)}
TRUE = {"depression": "phq", "anxiety": "gad"}


def auroc(score, y):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(np.asarray(score)).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(a, b):
    m = pd.notna(a) & pd.notna(b)
    return pd.Series(np.asarray(a)[m]).rank().corr(pd.Series(np.asarray(b)[m]).rank())


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ref", default="A")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()
runs = dict(r.split("=") for r in a.runs)

base = pd.read_pickle(os.path.join(a.base, "data", "clean", "baseline_clean.pkl")).set_index("interview__key")
test_csv = pd.read_csv(os.path.join(a.base, "data", "clean", "dam1", "test.csv"))
v6_people = set(test_csv.loc[test_csv.version == "v6", "user"])
sc = {k: pd.read_csv(os.path.join(a.results, f"{rid}_scores.csv")).query("split == 'test'") for k, rid in runs.items()}
# v6 test people scored by every run (a person can be missing if all their recordings were filtered out)
v6_people = v6_people.intersection(*[set(d.user) for d in sc.values()])
rng = np.random.default_rng(0)
v6_list = sorted(v6_people)
draws = [rng.integers(0, len(v6_list), len(v6_list)) for _ in range(a.boot)]

rows, paired, corr = [], [], []
person_v6 = {}
for k, rid in runs.items():
    d = sc[k]
    for task, cuts in CUTOFFS.items():
        t = d[d.task == task]
        p = t.groupby("user").agg(score=("score", "mean"), label=("label", "first"))
        for tset in ("own", "v6"):
            tt = t[t.user.isin(v6_people)] if tset == "v6" else t
            pp = p[p.index.isin(v6_people)] if tset == "v6" else p
            if tset == "v6":
                person_v6[(k, task)] = pp.reindex(v6_list)
            for c in cuts:
                for level, x in (("recording", tt), ("person", pp)):
                    val = auroc(x.score, x.label > c)
                    lo = hi = np.nan
                    if level == "person":  # bootstrap over people (person level only)
                        sv, yv = x.score.to_numpy(), (x.label > c).to_numpy()
                        bs = [auroc(sv[i], yv[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(a.boot))]
                        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                    users = x.user.unique() if level == "recording" else x.index
                    rows.append(dict(exp=k, run=rid, test_set=tset, task=task, cutoff=f">{c}", level=level,
                                     people=len(users), n=len(x), positives=int((x.label > c).sum()),
                                     auroc=val, ci_low=lo, ci_high=hi))
            # task-level person AUROC averaged over that task's cutoffs is in compare_auroc (mean over cutoffs)
        # correlations (own test set, person level)
        bb = base.reindex(p.index)
        age = bb.Q44_age.where(bb.Q44_age > 0)
        edu = bb.Q53_edu.where(bb.Q53_edu >= 0)
        corr.append(dict(exp=k, run=rid, task=task, people=len(p),
                         score_vs_true=spearman(p.score, p.label), score_vs_age=spearman(p.score, age),
                         score_vs_education=spearman(p.score, edu), true_vs_age=spearman(p.label, age),
                         true_vs_education=spearman(p.label, edu)))

for k in runs:
    if k == a.ref: continue
    for task, cuts in CUTOFFS.items():
        x, r = person_v6[(k, task)], person_v6[(a.ref, task)]
        for c in cuts:
            y = (r.label > c).to_numpy()
            diff = auroc(x.score, y) - auroc(r.score, y)
            bs = [auroc(x.score.to_numpy()[i], y[i]) - auroc(r.score.to_numpy()[i], y[i]) for i in draws]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            paired.append(dict(exp=k, ref=a.ref, task=task, cutoff=f">{c}", people=len(y), diff=diff,
                               ci_low=lo, ci_high=hi, p_le_0=float(np.mean(np.array(bs) <= 0))))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(a.results, "compare_auroc.csv"), index=False)
pd.DataFrame(paired).to_csv(os.path.join(a.results, "compare_paired.csv"), index=False)
pd.DataFrame(corr).to_csv(os.path.join(a.results, "compare_corr.csv"), index=False)
with pd.option_context("display.width", 220, "display.max_rows", 300, "display.precision", 3):
    print(out[out.level == "person"].pivot_table(index=["task", "cutoff"], columns=["test_set", "exp"], values="auroc"))
    print(pd.DataFrame(paired).to_string(index=False))
    print(pd.DataFrame(corr).to_string(index=False))
