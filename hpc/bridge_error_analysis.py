"""
bridge_error_analysis.py — where does DAM 1 fail? (run on the Mac: it reads the clean survey base).

Inputs
  --base    Bridge_baseline folder (reads data/clean/baseline_clean.pkl)
  --results folder with <run>_scores.csv and <run>_confusion.csv from hpc/bridge_eval.py
  --run     W&B run id
Outputs (in <results>):
  <run>_errors_by_group.csv     test errors per demographic group, per task and cutoff (person level)
  <run>_baseline_with_scores.pkl / .csv
                                the clean base + model scores per person (mean over the person's recordings) for
                                val/test people: <task>_score, <task>_split, and 0/1 predictions per cutoff
                                (<task>_pred_gt<c>) at the validation sn = sp threshold. Train people have no score.
Person level: each person's score is the mean of their recordings' scores; thresholds come from
<run>_confusion.csv (level == person).
"""
import argparse
import os

import numpy as np
import pandas as pd

GROUPS = {
    "gender": ("Q45_gender", {1: "male", 2: "female"}),
    "age": ("Q44_age", None),
    "mother_tongue": ("Q51_mothertongue", {1: "Havu", 2: "Shi", 3: "Tembo", 4: "Fuliru", 5: "Nyindu", 7: "Nyanga",
                                           8: "Bembe", 9: "Other"}),
    "displaced": ("LIST_Adult_Displaced", {0: "no", 1: "yes"}),
    "education": ("Q53_edu", None),
    "question_order": ("order_PHQ9GAD7", None),
}
MAIN = {"depression": (9, 14), "anxiety": (4, 9, 14)}
LABEL = {"depression": "phq", "anxiety": "gad"}


def group_values(base, name):
    col, labels = GROUPS[name]
    s = base[col]
    if name == "age":
        return pd.cut(s.where(s > 0), [17, 24, 34, 44, 120], labels=["18-24", "25-34", "35-44", "45+"]).astype(str)
    if name == "education":
        return pd.cut(s, [-1000, -1, 0, 7, 13, 30], labels=["dk/ref", "none", "primary", "secondary", "higher"]).astype(str)
    return s.map(labels) if labels else s.astype(str)


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--run", required=True)
a = ap.parse_args()

base = pd.read_pickle(os.path.join(a.base, "data", "clean", "baseline_clean.pkl")).set_index("interview__key")
sc = pd.read_csv(os.path.join(a.results, f"{a.run}_scores.csv"))
cm = pd.read_csv(os.path.join(a.results, f"{a.run}_confusion.csv"))
thr = {(r.task, int(r.cutoff[1:])): r.threshold for r in cm[cm.level == "person"].itertuples()}

person = sc.groupby(["task", "split", "user"]).agg(score=("score", "mean"), label=("label", "first")).reset_index()

# 1) base + scores (val/test people)
wide = base.copy()
for task in MAIN:
    p = person[person.task == task].set_index("user")
    wide[f"{task}_score"] = p.score
    wide[f"{task}_split"] = p.split
    for c in MAIN[task]:
        wide[f"{task}_pred_gt{c}"] = (p.score >= thr[(task, c)]).astype("Int64")
wide = wide.reset_index()
wide.to_pickle(os.path.join(a.results, f"{a.run}_baseline_with_scores.pkl"))
wide.to_csv(os.path.join(a.results, f"{a.run}_baseline_with_scores.csv"), index=False)

# 2) errors by group, test people only
rows = []
test = person[person.split == "test"]
for task, cuts in MAIN.items():
    t = test[test.task == task].set_index("user")
    for c in cuts:
        y = t.label > c
        pred = t.score >= thr[(task, c)]
        for gname in GROUPS:
            g = group_values(base.loc[t.index], gname)
            for val in sorted(g.unique()):
                m = (g == val).to_numpy()
                yy, pp = y[m], pred[m]
                tp, fp, fn, tn = int((pp & yy).sum()), int((pp & ~yy).sum()), int((~pp & yy).sum()), int((~pp & ~yy).sum())
                rows.append(dict(task=task, cutoff=f">{c}", group=gname, value=val, people=int(m.sum()),
                                 positives=tp + fn, TP=tp, FP=fp, FN=fn, TN=tn,
                                 sensitivity_recall=tp / (tp + fn) if tp + fn else np.nan,
                                 specificity=tn / (tn + fp) if tn + fp else np.nan,
                                 precision_ppv=tp / (tp + fp) if tp + fp else np.nan,
                                 false_negative_rate=fn / (tp + fn) if tp + fn else np.nan,
                                 false_positive_rate=fp / (tn + fp) if tn + fp else np.nan))
err = pd.DataFrame(rows)
err.to_csv(os.path.join(a.results, f"{a.run}_errors_by_group.csv"), index=False)
with pd.option_context("display.width", 220, "display.max_rows", 200, "display.precision", 2):
    print(err[err.cutoff.isin([">9"])].drop(columns=["TP", "FP", "FN", "TN"]).to_string(index=False))
print(f"\nwrote {a.run}_baseline_with_scores.(pkl|csv): {wide.depression_score.notna().sum()} people with scores")
