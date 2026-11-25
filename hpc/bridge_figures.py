"""
bridge_figures.py — figures comparing DAM 1 runs A-E on BRIDGE-MH (run on the Mac: it reads the survey base).

Privacy rule: every figure and every printed number is an aggregate. No individual points are drawn (no scatter of
people), boxplots hide outliers, any group with fewer than MIN_N people is not shown, and nothing with an
interview key is printed. Individual-level data is only read in memory on the Mac.

Usage (from KitsugiCode/):
    python3 kintsugi-hpc-uniandes/hpc/bridge_figures.py --base Bridge_baseline \
        --results Bridge_baseline/data/clean/results --out Avance_25_NOV_figs
Writes <out>/*.png and <out>/summary.json (aggregated numbers used in the slide text).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = dict(A="pnybari4", B="bnkqcsas", C="qviai5hj", D="oxmqyl0h", E="jmy587yu")
NAME = dict(A="A · all · small.en", B="B · all · small multi.", C="C · all · large-v2",
            D="D · v6 · small.en", E="E · v6 · large-v2")
COL = dict(A="#0089B8", B="#8FA3B8", C="#002060", D="#1E7D4F", E="#D94F2B")
CUTS = {"depression": (9, 14), "anxiety": (4, 9, 14)}
TRUE = {"depression": "phq", "anxiety": "gad"}
MIN_N = 10
NAVY, GREY, GRID = "#002060", "#555F6D", "#D9DEE6"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": NAVY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.titlecolor": NAVY, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=500)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
S = {}  # summary numbers (aggregates only)


def save(fig, name):
    fig.savefig(os.path.join(a.out, name + ".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, bool)
    m = ~np.isnan(score); score, y = score[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, y, n=None):
    score = np.asarray(score, float); y = np.asarray(y, bool); n = n or a.boot
    bs = [auroc(score[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.nanpercentile(bs, [2.5, 97.5])


def spearman(x, y):
    m = pd.notna(x) & pd.notna(y)
    return pd.Series(np.asarray(x, float)[m]).rank().corr(pd.Series(np.asarray(y, float)[m]).rank())


def roc(score, y):
    o = np.argsort(-np.asarray(score)); y = np.asarray(y, bool)[o]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()]); fpr = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
    return fpr, tpr


def logit_fit(X, y, l2=1e-3, it=50):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); Wd = p * (1 - p)
        H = X.T @ (X * Wd[:, None]) + l2 * np.eye(len(w)); g = X.T @ (y - p) - l2 * w
        w += np.linalg.solve(H, g)
    return w


def logit_pred(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


# ------------------------------------------------------------------ data
base = pd.read_pickle(os.path.join(a.base, "data", "clean", "baseline_clean.pkl")).set_index("interview__key")
csv = pd.concat([pd.read_csv(os.path.join(a.base, "data", "clean", "dam1", f"{s}.csv")).assign(split=s)
                 for s in ("train", "val", "test")], ignore_index=True)
csv["duration"] = csv["prompt_vad_trimmed_duration"].fillna(csv["prompt_vad_cut_duration"]) \
    if "prompt_vad_trimmed_duration" in csv else csv["prompt_vad_cut_duration"]
csv["section"] = np.where(csv.uuid.str.contains("_f__"), "F", "A")
people = csv.groupby("user").agg(version=("version", "first"), split=("split", "first"), n_rec=("uuid", "size"),
                                 duration=("duration", "mean"), phq=("phq", "first"), gad=("gad", "first"))
demo = base.reindex(people.index)
people["age"] = demo.Q44_age.where(demo.Q44_age > 0)
people["edu_years"] = demo.Q53_edu.where(demo.Q53_edu >= 0)
people["female"] = (demo.Q45_gender == 2).astype(float).where(demo.Q45_gender.isin([1, 2]))
people["displaced"] = demo.LIST_Adult_Displaced.astype(float)
people["age_g"] = pd.cut(people.age, [17, 24, 34, 44, 120], labels=["18–24", "25–34", "35–44", "45+"]).astype(str)
people["edu_g"] = pd.cut(people.edu_years, [-1, 0, 7, 13, 30], labels=["none", "primary", "secondary", "higher"]).astype(str)
people["gender_g"] = people.female.map({1.0: "women", 0.0: "men"})
people["displaced_g"] = people.displaced.map({1.0: "displaced", 0.0: "not displaced"})
mt = {1: "Havu", 2: "Shi", 3: "Tembo", 4: "Fuliru", 5: "Nyindu", 7: "Nyanga", 8: "Bembe", 9: "Other"}
people["tongue_g"] = demo.Q51_mothertongue.map(mt)
interv_col = next((c for c in base.columns if c.lower() in ("interviewer", "interviewer_id", "enumerator", "enum_id",
                                                         "interviewer__id", "responsible", "enumerator_id")), None)
if interv_col is None:
    cands = [c for c in base.columns if any(k in c.lower() for k in ("interv", "enum", "responsible"))]
    interv_col = cands[0] if cands else None
people["interviewer"] = demo[interv_col].astype(str) if interv_col else np.nan
S["interviewer_column"] = interv_col
V6 = people.version == "v6"

scores = {k: pd.read_csv(os.path.join(a.results, f"{r}_scores.csv")) for k, r in RUNS.items()}
conf = {k: pd.read_csv(os.path.join(a.results, f"{r}_confusion.csv")) for k, r in RUNS.items()}


def person_scores(k, task, split="test"):
    d = scores[k]; d = d[(d.task == task) & (d.split == split)]
    agg = {"score": ("score", "mean"), "label": ("label", "first")}
    if f"prob_gt{CUTS[task][0]}" in d:
        for c in CUTS[task]: agg[f"p{c}"] = (f"prob_gt{c}", "mean")
    return d.groupby("user").agg(**agg)


common = set(people.index[V6 & (people.split == "test")])
for k in RUNS: common &= set(person_scores(k, "depression").index)
common = sorted(common)
S["common_v6_test_people"] = len(common)

# ================================================================== 1. versions: data profile
ver = people.groupby("version").agg(people=("n_rec", "size"), rec_per_person=("n_rec", "mean"),
                                     duration=("duration", "median"), phq9=("phq", lambda s: (s > 9).mean()),
                                     gad9=("gad", lambda s: (s > 9).mean()), phq_mean=("phq", "mean"),
                                     female=("female", "mean"), age=("age", "median"), displaced=("displaced", "mean"),
                                     edu=("edu_years", "median"))
S["versions"] = ver.round(3).reset_index().to_dict("records")
fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
panels = [("people", "People", "{:.0f}"), ("rec_per_person", "Recordings per person", "{:.2f}"),
          ("phq9", "PHQ > 9 (share)", "{:.0%}"), ("gad9", "GAD > 9 (share)", "{:.0%}"),
          ("age", "Median age (years)", "{:.0f}"), ("displaced", "Displaced (share)", "{:.0%}")]
for axi, (col, t, fmt) in zip(ax.ravel(), panels):
    c = ["#1E7D4F" if v == "v6" else "#8FA3B8" for v in ver.index]
    b = axi.bar(ver.index, ver[col], color=c, width=0.65)
    for rect, v in zip(b, ver[col]): axi.text(rect.get_x() + rect.get_width() / 2, v, fmt.format(v), ha="center",
                                              va="bottom", fontsize=9, color=NAVY)
    axi.set_title(t); axi.set_ylim(0, ver[col].max() * 1.2); axi.grid(axis="x", visible=False)
fig.suptitle("Questionnaire versions: who is in each (green = v6)", color=NAVY, fontweight="bold", y=1.01)
fig.tight_layout(); save(fig, "01_versions_profile")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for axi, (lab, mx) in zip(ax, (("phq", 27), ("gad", 21))):
    bins = np.arange(-0.5, mx + 1.5, 1)
    for grp, m, c in (("v1–v5", ~V6, "#8FA3B8"), ("v6", V6, "#1E7D4F")):
        axi.hist(people.loc[m, lab], bins=bins, density=True, histtype="step", lw=2.2, color=c,
                 label=f"{grp} ({m.sum():,} people)")
    axi.axvline(9.5, color=GREY, ls="--", lw=1); axi.set_xlabel(f"{lab.upper()} total"); axi.set_ylabel("share of people")
    axi.set_title(f"{lab.upper()} distribution"); axi.legend()
fig.tight_layout(); save(fig, "02_label_distribution_v6_vs_rest")

# protocol: recording sections in v6
sec = csv[csv.version == "v6"].groupby("section").agg(recordings=("uuid", "size"), duration=("duration", "median"))
S["v6_sections"] = sec.round(1).reset_index().to_dict("records")
S["dur_by_group"] = {"v1-v5": float(csv[csv.version != "v6"].duration.median()), "v6": float(csv[csv.version == "v6"].duration.median())}

# ================================================================== 2. AUROC with CIs, own test and common v6
rows = []
for k in RUNS:
    for task, cuts in CUTS.items():
        p = person_scores(k, task)
        for tset, pp in (("own", p), ("v6", p.reindex(common))):
            for c in cuts:
                y = pp.label > c; lo, hi = boot_ci(pp.score, y)
                rows.append(dict(exp=k, test_set=tset, task=task, cut=f"{TRUE[task].upper()} > {c}", auroc=auroc(pp.score, y),
                                 lo=lo, hi=hi, n=len(pp), pos=int(y.sum())))
AU = pd.DataFrame(rows); S["auroc"] = AU.round(3).to_dict("records")
for tset, title in (("own", "Each run's own test set (A–C: 664 people, D–E: 272)"),
                    ("v6", f"Same {len(common)} v6 test people for every run")):
    d = AU[AU.test_set == tset]; cuts = list(dict.fromkeys(d.cut)); x = np.arange(len(cuts)); w = 0.16
    fig, axi = plt.subplots(figsize=(13, 5))
    for i, k in enumerate(RUNS):
        dk = d[d.exp == k].set_index("cut").loc[cuts]
        axi.bar(x + (i - 2) * w, dk.auroc, w * 0.92, color=COL[k], label=NAME[k],
                yerr=[dk.auroc - dk.lo, dk.hi - dk.auroc], error_kw=dict(ecolor=GREY, lw=0.9, capsize=2))
    axi.axhline(0.5, color=GREY, lw=1, ls=":"); axi.set_xticks(x, [f"{c}\n({d[d.cut == c].pos.iloc[-1]} pos.)" for c in cuts])
    axi.set_ylim(0.3, 0.95); axi.set_ylabel("test AUROC (person level)")
    axi.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9.5); axi.grid(axis="x", visible=False)
    save(fig, f"03_auroc_{tset}")

# paired differences vs A on common people
pr = []
for task, cuts in CUTS.items():
    P = {k: person_scores(k, task).reindex(common) for k in RUNS}
    y_all = {c: (P["A"].label > c).to_numpy() for c in cuts}
    draws = [rng.integers(0, len(common), len(common)) for _ in range(a.boot)]
    for k in "BCDE":
        for c in cuts:
            y = y_all[c]; s1, s0 = P[k].score.to_numpy(), P["A"].score.to_numpy()
            bs = [auroc(s1[i], y[i]) - auroc(s0[i], y[i]) for i in draws]
            pr.append(dict(exp=k, cut=f"{TRUE[task].upper()} > {c}", diff=auroc(s1, y) - auroc(s0, y),
                           lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
PR = pd.DataFrame(pr); S["paired_vs_A"] = PR.round(3).to_dict("records")
cuts = list(dict.fromkeys(PR.cut))
fig, axi = plt.subplots(figsize=(13, 5))
for i, k in enumerate("BCDE"):
    dk = PR[PR.exp == k].set_index("cut").loc[cuts]; yy = np.arange(len(cuts)) + (i - 1.5) * 0.18
    axi.errorbar(dk["diff"], yy, xerr=[dk["diff"] - dk.lo, dk.hi - dk["diff"]], fmt="o", color=COL[k], ms=7, lw=2,
                 capsize=3, label=NAME[k] + " − A")
axi.axvline(0, color=GREY, lw=1.2); axi.set_yticks(np.arange(len(cuts)), cuts); axi.invert_yaxis()
axi.set_xlabel("AUROC difference vs. run A (same people, 95 % bootstrap CI)")
axi.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9.5)
axi.grid(axis="y", visible=False); save(fig, "04_paired_vs_A")

# ROC curves on common people
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
for axi, (task, c) in zip(ax, (("depression", 9), ("anxiety", 4), ("anxiety", 9))):
    for k in RUNS:
        p = person_scores(k, task).reindex(common); y = p.label > c
        f, t = roc(p.score, y); axi.plot(f, t, color=COL[k], lw=2, label=f"{k}: {auroc(p.score, y):.2f}")
    axi.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1); axi.set_aspect("equal")
    axi.set_xlabel("1 − specificity"); axi.set_ylabel("sensitivity"); axi.set_title(f"{TRUE[task].upper()} > {c}")
    axi.legend(loc="lower right", fontsize=9, title="AUROC", title_fontsize=9)
fig.suptitle(f"ROC curves, same {len(common)} v6 test people", color=NAVY, fontweight="bold")
fig.tight_layout(); save(fig, "05_roc_common")

# ================================================================== 3. training history
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
S["history"] = {}
for k, r in RUNS.items():
    h = pd.read_csv(os.path.join(a.results, f"{r}_val.csv"))
    au = h["metrics/val/auroc@0%i"].to_numpy(); vl = (h["anxiety/metrics/val/loss_exact22"] + h["depression/metrics/val/loss_exact28"]).to_numpy()
    best = int(h["metrics/val/sn_eq_sp@0%i"].values.argmax())
    S["history"][k] = dict(validations=len(h), best=best + 1)
    x = np.arange(1, len(au) + 1)
    ax[0].plot(x, au, color=COL[k], lw=1.8, label=NAME[k]); ax[0].plot(best + 1, au[best], "o", color=COL[k], ms=8, mec="white")
    ax[1].plot(x, vl, color=COL[k], lw=1.8, label=NAME[k])
ax[0].set_title("Validation AUROC (dot = selected checkpoint)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=9)
ax[1].set_title("Validation loss (depression + anxiety)"); ax[1].set_xlabel("epoch")
fig.tight_layout(); save(fig, "06_training_history")

# ================================================================== 4. confusion and threshold metrics
fig, ax = plt.subplots(1, 5, figsize=(13, 3.3))
for axi, k in zip(ax, RUNS):
    r = conf[k]; r = r[(r.level == "person") & (r.task == "depression") & (r.cutoff == ">9")].iloc[0]
    m = np.array([[r.TP, r.FN], [r.FP, r.TN]]); mn = m / m.sum(1, keepdims=True)
    axi.imshow(mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axi.text(j, i, f"{m[i, j]}\n{mn[i, j]:.0%}", ha="center", va="center", fontsize=11,
                     color="white" if mn[i, j] > 0.55 else NAVY)
    axi.set_xticks([0, 1], ["flagged", "not flagged"]); axi.set_yticks([0, 1], ["PHQ > 9", "PHQ ≤ 9"])
    axi.set_title(NAME[k], fontsize=10.5, color=COL[k]); axi.grid(False)
    for sp in axi.spines.values(): sp.set_visible(False)
fig.suptitle("Confusion matrices, PHQ > 9, test people (row %)", color=NAVY, fontweight="bold", y=1.04)
fig.tight_layout(); save(fig, "07_confusion_phq9")

met = pd.concat([c[c.level == "person"].assign(exp=k) for k, c in conf.items()])
met["cut"] = met.task.map(lambda t: TRUE[t].upper()) + " " + met.cutoff.str.replace(">", "> ")
cuts = list(dict.fromkeys(met.cut))
fig, ax = plt.subplots(2, 2, figsize=(13, 6.6), sharex=True)
for axi, (col, t) in zip(ax.ravel(), (("sensitivity_recall", "Recall (sensitivity)"), ("specificity", "Specificity"),
                                      ("precision_ppv", "Precision (PPV)"), ("f1", "F1"))):
    x = np.arange(len(cuts)); w = 0.16
    for i, k in enumerate(RUNS):
        dk = met[met.exp == k].set_index("cut").loc[cuts]
        axi.bar(x + (i - 2) * w, dk[col], w * 0.92, color=COL[k], label=NAME[k])
    axi.set_title(t); axi.set_ylim(0, 1); axi.set_xticks(x, cuts); axi.grid(axis="x", visible=False)
ax[0, 0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.3), fontsize=9.5)
fig.tight_layout(); save(fig, "08_threshold_metrics")

# ================================================================== 5. model responses
# score by true severity band (boxplots without outliers)
bands = [(0, 4, "0–4"), (5, 9, "5–9"), (10, 14, "10–14"), (15, 19, "15–19"), (20, 27, "20–27")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
S["score_by_band"] = {}
for axi, k in zip(ax, ("A", "D")):
    p = person_scores(k, "depression")
    data = [p.score[(p.label >= lo) & (p.label <= hi)].to_numpy() for lo, hi, _ in bands]
    bp = axi.boxplot(data, showfliers=False, patch_artist=True, widths=0.55, medianprops=dict(color="white", lw=2))
    for b in bp["boxes"]: b.set_facecolor(COL[k]); b.set_edgecolor(COL[k])
    axi.set_xticks(range(1, 6), [f"{l}\n(n={len(d)})" for (_, _, l), d in zip(bands, data)])
    axi.set_xlabel("true PHQ-9 band"); axi.set_ylabel("model score (person)"); axi.set_title(NAME[k])
    S["score_by_band"][k] = [float(np.median(d)) if len(d) else None for d in data]
fig.suptitle("Model score rises with true severity, with heavy overlap", color=NAVY, fontweight="bold")
fig.tight_layout(); save(fig, "09_score_by_phq_band")

# calibration of P(PHQ > 9)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for axi, (task, c) in zip(ax, (("depression", 9), ("anxiety", 9))):
    for k in RUNS:
        p = person_scores(k, task)
        if f"p{c}" not in p: continue
        q = pd.qcut(p[f"p{c}"], 5, duplicates="drop")
        g = p.groupby(q, observed=True).agg(pred=(f"p{c}", "mean"), obs=("label", lambda s: (s > c).mean()))
        axi.plot(g.pred, g.obs, "o-", color=COL[k], lw=2, label=NAME[k])
    axi.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1); axi.set_xlim(0, 0.8); axi.set_ylim(0, 0.8)
    axi.set_xlabel(f"predicted P({TRUE[task].upper()} > {c}) (quintile mean)"); axi.set_ylabel("observed share")
    axi.set_title(f"Calibration, {TRUE[task].upper()} > {c}")
ax[0].legend(fontsize=9); fig.tight_layout(); save(fig, "10_calibration")

# within-person agreement of the two v6 recordings (section A vs F), run D
d = scores["D"]; d = d[(d.task == "depression") & (d.split == "test")].copy()
d["section"] = np.where(d.uuid.str.contains("_f__"), "F", "A")
wide = d.pivot_table(index="user", columns="section", values="score", aggfunc="mean").dropna()
lab = d.groupby("user").label.first().reindex(wide.index)
S["v6_sections_D"] = dict(people=len(wide), spearman_A_vs_F=spearman(wide.A, wide.F),
                          auroc_A=auroc(wide.A, lab > 9), auroc_F=auroc(wide.F, lab > 9),
                          auroc_mean=auroc(wide.mean(1), lab > 9))

# ================================================================== 6. demographics
GROUPS = [("age_g", "Age", ["18–24", "25–34", "35–44", "45+"]), ("edu_g", "Education", ["none", "primary", "secondary", "higher"]),
          ("gender_g", "Gender", ["women", "men"]), ("displaced_g", "Displacement", ["not displaced", "displaced"]),
          ("tongue_g", "Mother tongue", ["Shi", "Havu", "Tembo", "Other"])]


def zmean_by_group(k, task, gcol, order):
    p = person_scores(k, task).join(people[[gcol]])
    zs = (p.score - p.score.mean()) / p.score.std(); zt = (p.label - p.label.mean()) / p.label.std()
    out = []
    for g in order:
        m = (p[gcol] == g).to_numpy()
        if m.sum() < MIN_N: continue
        se_s, se_t = zs[m].std() / np.sqrt(m.sum()), zt[m].std() / np.sqrt(m.sum())
        out.append(dict(group=g, n=int(m.sum()), score=zs[m].mean(), score_se=se_s, true=zt[m].mean(), true_se=se_t))
    return pd.DataFrame(out)


for k in ("A", "D"):
    fig, ax = plt.subplots(1, 5, figsize=(13, 4.2), sharey=True)
    for axi, (gcol, t, order) in zip(ax, GROUPS):
        z = zmean_by_group(k, "depression", gcol, order); x = np.arange(len(z))
        axi.errorbar(x - 0.1, z.true, yerr=1.96 * z.true_se, fmt="s", color=GREY, ms=7, capsize=3, label="true PHQ-9")
        axi.errorbar(x + 0.1, z.score, yerr=1.96 * z.score_se, fmt="o", color=COL[k], ms=7, capsize=3, label="model score")
        axi.axhline(0, color=GREY, lw=0.8); axi.set_xticks(x, [f"{g}\n{n}" for g, n in zip(z.group, z.n)], fontsize=9)
        axi.set_title(t); axi.grid(axis="x", visible=False)
    ax[0].set_ylabel("mean, standardised (z)"); ax[0].legend(fontsize=9, loc="upper left")
    fig.suptitle(f"Run {NAME[k]}: model score vs. true PHQ-9 by group (test people, 95 % CI)", color=NAVY, fontweight="bold")
    fig.tight_layout(); save(fig, f"11_demo_means_{k}")

# AUROC within groups (PHQ > 9)
sub = []
for k in ("A", "C", "D"):
    p = person_scores(k, "depression").join(people[[g for g, _, _ in GROUPS]])
    for gcol, t, order in GROUPS:
        for g in order:
            m = (p[gcol] == g).to_numpy(); y = (p.label > 9).to_numpy()[m]
            if m.sum() < MIN_N or y.sum() < 5 or (~y).sum() < 5: continue
            lo, hi = boot_ci(p.score.to_numpy()[m], y)
            sub.append(dict(exp=k, group=t, value=g, n=int(m.sum()), pos=int(y.sum()), auroc=auroc(p.score.to_numpy()[m], y), lo=lo, hi=hi))
    sub.append(dict(exp=k, group="All", value="all test", n=len(p), pos=int((p.label > 9).sum()),
                    auroc=auroc(p.score, p.label > 9), lo=np.nan, hi=np.nan))
SUB = pd.DataFrame(sub); S["auroc_within_groups"] = SUB.round(3).to_dict("records")
labels = list(dict.fromkeys(zip(SUB.group, SUB.value)))
fig, axi = plt.subplots(figsize=(13, 5.6))
for i, k in enumerate(("A", "C", "D")):
    dk = SUB[SUB.exp == k].set_index(["group", "value"]).reindex(labels)
    yy = np.arange(len(labels)) + (i - 1) * 0.22
    axi.errorbar(dk.auroc, yy, xerr=[dk.auroc - dk.lo, dk.hi - dk.auroc], fmt="o", color=COL[k], ms=6, capsize=2, lw=1.4, label=NAME[k])
axi.axvline(0.5, color=GREY, ls=":", lw=1)
axi.set_yticks(np.arange(len(labels)), [f"{g}: {v}" for g, v in labels], fontsize=9.5); axi.invert_yaxis()
axi.set_xlabel("AUROC for PHQ > 9 inside the group (95 % bootstrap CI)"); axi.set_xlim(0.2, 1.0)
axi.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
axi.grid(axis="y", visible=False); save(fig, "12_auroc_within_groups")

# recall / specificity by group heatmap (PHQ > 9, all runs)
eg = []
for k, r in RUNS.items():
    e = pd.read_csv(os.path.join(a.results, f"{r}_errors_by_group.csv"))
    e = e[(e.task == "depression") & (e.cutoff == ">9") & (e.group.isin(["age", "education", "gender"])) & (e.people >= MIN_N)]
    e = e[e.value != "dk/ref"]
    eg.append(e.assign(exp=k))
EG = pd.concat(eg); EG["row"] = EG.group + ": " + EG.value.astype(str)
order = ["age: 18-24", "age: 25-34", "age: 35-44", "age: 45+", "education: none", "education: primary",
         "education: secondary", "education: higher", "gender: female", "gender: male"]
order = [o for o in order if o in set(EG.row)]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for axi, (col, t) in zip(ax, (("sensitivity_recall", "Recall (missed cases = 1 − recall)"), ("specificity", "Specificity (false alarms = 1 − spec.)"))):
    M = EG.pivot_table(index="row", columns="exp", values=col).reindex(order)[list(RUNS)]
    axi.imshow(M.values, cmap="RdYlGn", vmin=0.2, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if pd.notna(v): axi.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=NAVY)
    axi.set_xticks(range(5), list(RUNS)); axi.set_yticks(range(len(order)), order); axi.set_title(t); axi.grid(False)
fig.suptitle("PHQ > 9 at the validation threshold, by group (test people)", color=NAVY, fontweight="bold")
fig.tight_layout(); save(fig, "13_recall_spec_by_group")

# correlations: score vs true, age, education, female, displaced; partial correlation controlling demographics
cr = []
for k in RUNS:
    for task in CUTS:
        p = person_scores(k, task).join(people[["age", "edu_years", "female", "displaced"]])
        row = dict(exp=k, task=task, n=len(p), true=spearman(p.score, p.label), age=spearman(p.score, p.age),
                   education=spearman(p.score, p.edu_years), female=spearman(p.score, p.female),
                   displaced=spearman(p.score, p.displaced))
        q = p.dropna()
        R = q[["score", "label", "age", "edu_years", "female", "displaced"]].rank()
        X = np.column_stack([np.ones(len(R)), R[["age", "edu_years", "female", "displaced"]].values])
        res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        row["true_partial"] = np.corrcoef(res(R.score.values), res(R.label.values))[0, 1]
        cr.append(row)
CR = pd.DataFrame(cr); S["correlations"] = CR.round(3).to_dict("records")
ref = {}
for task in CUTS:
    p = people.loc[people.split == "test"]
    ref[task] = dict(age=spearman(p[TRUE[task]], p.age), education=spearman(p[TRUE[task]], p.edu_years),
                     female=spearman(p[TRUE[task]], p.female), displaced=spearman(p[TRUE[task]], p.displaced))
S["true_label_correlations_test"] = {t: {kk: round(v, 3) for kk, v in d.items()} for t, d in ref.items()}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
cols = ["true", "true_partial", "age", "education", "female", "displaced"]
names = ["true label", "true label\n(net of demo.)", "age", "education", "female", "displaced"]
for axi, task in zip(ax, CUTS):
    x = np.arange(len(cols)); w = 0.14
    for i, k in enumerate(RUNS):
        dk = CR[(CR.exp == k) & (CR.task == task)].iloc[0]
        axi.bar(x + (i - 2.5) * w, [dk[c] for c in cols], w * 0.92, color=COL[k], label=NAME[k])
    axi.bar(x + 2.5 * w, [np.nan, np.nan] + [ref[task][c] for c in cols[2:]], w * 0.92, color="#C2CBD6", label="true label itself")
    axi.axhline(0, color=GREY, lw=0.8); axi.set_xticks(x, names, fontsize=9.5); axi.set_title(f"{task.capitalize()} score: Spearman correlation")
    axi.grid(axis="x", visible=False)
ax[0].legend(ncol=3, fontsize=8.5, loc="lower left"); fig.tight_layout(); save(fig, "14_correlations")

# ================================================================== 7. demographics-only baseline and combination
def design(idx):
    q = people.reindex(idx)
    cols = {"age": q.age.fillna(people.age.median()), "edu": q.edu_years.fillna(people.edu_years.median()),
            "female": q.female.fillna(1), "displaced": q.displaced.fillna(0)}
    for v in ("v1", "v2", "v4", "v5"): cols[v] = (q.version == v).astype(float)
    X = pd.DataFrame(cols)
    return ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).to_numpy()


bl = []
for task, c in (("depression", 9), ("anxiety", 9)):
    tr = people.index[people.split == "train"]; yt = (people.loc[tr, TRUE[task]] > c).to_numpy().astype(float)
    w = logit_fit(design(tr), yt)
    for tset, idx in (("all", people.index[people.split == "test"]), ("v6", pd.Index(common))):
        demo_s = logit_pred(w, design(idx)); y = (people.loc[idx, TRUE[task]] > c).to_numpy()
        lo, hi = boot_ci(demo_s, y)
        bl.append(dict(task=task, cut=f"{TRUE[task].upper()} > {c}", test_set=tset, model="demographics only",
                       auroc=auroc(demo_s, y), lo=lo, hi=hi))
        for k in (("A", "C") if tset == "all" else ("A", "D")):
            # voice only
            p = person_scores(k, task).reindex(idx); lo, hi = boot_ci(p.score, y)
            bl.append(dict(task=task, cut=f"{TRUE[task].upper()} > {c}", test_set=tset, model=f"voice {k}",
                           auroc=auroc(p.score, y), lo=lo, hi=hi))
            # voice + demographics, combined on validation people, tested on test people
            pv = person_scores(k, task, "val"); vidx = pv.index
            Xv = np.column_stack([design(vidx), (pv.score - pv.score.mean()) / pv.score.std()])
            wv = logit_fit(Xv, (pv.label > c).to_numpy().astype(float))
            Xt = np.column_stack([design(idx), (p.score - pv.score.mean()) / pv.score.std()])
            comb = logit_pred(wv, Xt); lo, hi = boot_ci(comb, y)
            bl.append(dict(task=task, cut=f"{TRUE[task].upper()} > {c}", test_set=tset, model=f"voice {k} + demographics",
                           auroc=auroc(comb, y), lo=lo, hi=hi))
BL = pd.DataFrame(bl); S["baselines"] = BL.round(3).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for axi, tset in zip(ax, ("all", "v6")):
    d = BL[BL.test_set == tset]; models = list(dict.fromkeys(d.model)); cuts = list(dict.fromkeys(d.cut))
    x = np.arange(len(cuts)); w = 0.8 / len(models)
    pal = {"demographics only": "#C2CBD6"}
    for i, mname in enumerate(models):
        dk = d[d.model == mname].set_index("cut").loc[cuts]
        k = mname.split()[1] if mname.startswith("voice") else None
        color = pal.get(mname, COL.get(k, GREY)); hatch = "//" if "+" in mname else None
        bars = axi.bar(x + (i - (len(models) - 1) / 2) * w, dk.auroc, w * 0.9, color=color if not hatch else "white",
                       edgecolor=color, hatch=hatch, lw=1.5, label=mname, yerr=[dk.auroc - dk.lo, dk.hi - dk.auroc],
                       error_kw=dict(ecolor=GREY, lw=0.9, capsize=2))
        for b_, v, top in zip(bars, dk.auroc, dk.hi): axi.text(b_.get_x() + b_.get_width() / 2, top + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=NAVY)
    axi.axhline(0.5, color=GREY, ls=":", lw=1); axi.set_ylim(0.4, 1.0); axi.set_xticks(x, cuts)
    axi.set_title("All-versions test (664 people)" if tset == "all" else f"v6 test ({len(common)} people)")
    axi.legend(fontsize=8.5, loc="upper left", ncol=2); axi.grid(axis="x", visible=False)
ax[0].set_ylabel("test AUROC (person level)")
fig.suptitle("How much does the voice add to what demographics already predict?", color=NAVY, fontweight="bold")
fig.tight_layout(); save(fig, "15_demographic_baseline")

# ================================================================== 8. agreement between runs, interviewer
M = pd.DataFrame({k: person_scores(k, "depression").reindex(common).score for k in RUNS})
C = M.rank().corr(); S["run_agreement"] = C.round(3).to_dict()
fig, axi = plt.subplots(figsize=(5.6, 4.8))
axi.imshow(C.values, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5): axi.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", color="white" if C.values[i, j] > 0.6 else NAVY)
axi.set_xticks(range(5), list(RUNS)); axi.set_yticks(range(5), list(RUNS)); axi.grid(False)
axi.set_title(f"Do the runs agree? Spearman between\ndepression scores ({len(common)} v6 test people)", fontsize=11)
save(fig, "16_run_agreement")

if interv_col:
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k in RUNS:
        p = person_scores(k, "depression").join(people[["interviewer"]])
        p = p[p.groupby("interviewer").score.transform("size") >= 5]
        g = p.groupby("interviewer").agg(n=("score", "size"), score=("score", "mean"), true=("label", "mean"))
        share = lambda v: float(p.groupby("interviewer")[v].transform("mean").var() / p[v].var())
        sc_w = p.score - p.groupby("interviewer").score.transform("mean")
        y = p.label > 9
        rows.append(dict(exp=k, people=len(p), interviewers=len(g), spearman_interviewer_means=spearman(g.score, g.true),
                         score_var_between=share("score"), label_var_between=share("label"),
                         auroc=auroc(p.score, y), auroc_within_interviewer=auroc(sc_w, y)))
    IV = pd.DataFrame(rows); S["interviewer"] = IV.round(3).to_dict("records")
    p = person_scores("A", "depression").join(people[["interviewer"]])
    g = p.groupby("interviewer").agg(n=("score", "size"), score=("score", "mean"), true=("label", "mean"))
    g = g[g.n >= MIN_N]
    ax[0].scatter(g.true, g.score, s=g.n * 4, color=COL["A"], alpha=0.7, edgecolor="white")
    ax[0].set_xlabel("interviewer's mean true PHQ-9"); ax[0].set_ylabel("interviewer's mean model score")
    ax[0].set_title(f"Run A: one bubble per interviewer ({len(g)} with >= {MIN_N} test people)")
    x = np.arange(len(IV)); w = 0.38
    ax[1].bar(x - w / 2, IV.auroc, w * 0.92, color=[COL[k] for k in IV.exp], label="as is")
    ax[1].bar(x + w / 2, IV.auroc_within_interviewer, w * 0.92, color="white", edgecolor=[COL[k] for k in IV.exp],
              hatch="//", lw=1.5, label="within interviewer")
    for xi, (v1, v2) in enumerate(zip(IV.auroc, IV.auroc_within_interviewer)):
        ax[1].text(xi - w / 2, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=9, color=NAVY)
        ax[1].text(xi + w / 2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=9, color=NAVY)
    ax[1].set_xticks(x, IV.exp); ax[1].set_ylim(0.4, 0.8); ax[1].axhline(0.5, color=GREY, ls=":", lw=1)
    ax[1].set_title("PHQ > 9 AUROC: as is vs. ranking only within each interviewer"); ax[1].legend(fontsize=9)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "17_interviewer")

with open(os.path.join(a.out, "summary.json"), "w") as f:
    json.dump(S, f, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
print(json.dumps(S, indent=1, default=str))
