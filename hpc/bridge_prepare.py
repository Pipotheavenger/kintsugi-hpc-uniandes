"""
bridge_prepare.py — turn the BRIDGE-MH baseline (Bukavu) into DAM 1 inputs. Runs on the Mac.

Reads Bridge_baseline/data/raw (never modified) and data/clean/baseline_clean.pkl (built by
Bridge_baseline/scripts/01_build_clean.ipynb). Writes to <out>:
    audio/<interview__key>/<AudioN__ | AudioN_f__>.wav   16 kHz, mono, 16-bit PCM (only in_analysis == 1)
    train.csv, val.csv, test.csv                          DAM 1 data contract, no personal data

Steps (idempotent, can be re-run until done):
    python hpc/bridge_prepare.py audio --base <Bridge_baseline> --out <out> [--budget 150]
    python hpc/bridge_prepare.py csv   --base <Bridge_baseline> --out <out>

CSV columns: uuid (one per recording), filename (<key>/<stem>, no extension), user (interview__key,
lets KintsugiMetadata merge both recordings of a person in train), phq, gad, version, duration in
`prompt_vad_trimmed_duration` (train) / `prompt_vad_cut_duration` (val, test).
Split: 70/15/15 by person, stratified by questionnaire version and PHQ > 9, seed 0.
"""
import argparse
import os
import subprocess
import tempfile
import time
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import pandas as pd

DURATION_COLUMN = {"train": "prompt_vad_trimmed_duration", "val": "prompt_vad_cut_duration",
                   "test": "prompt_vad_cut_duration"}


def analysis_people(base):
    df = pd.read_pickle(os.path.join(base, "data", "clean", "baseline_clean.pkl"))
    return df[df.in_analysis == 1][["interview__key", "phq", "gad", "version"]]


def convert(args):
    keys = set(analysis_people(args.base).interview__key)
    out_audio = os.path.join(args.out, "audio")
    todo = []
    for z in sorted(glob(os.path.join(args.base, "data", "raw", "WB_Survey_*_Binary_*.zip"))):
        for name in zipfile.ZipFile(z).namelist():
            key, fn = name.split("/")
            dst = os.path.join(out_audio, key, fn.replace(".m4a", ".wav"))
            if key in keys and not os.path.exists(dst):
                todo.append((z, name, dst))
    print(f"to convert: {len(todo)}", flush=True)

    t0 = time.time()
    tmp = tempfile.mkdtemp()

    def one(item):
        z, name, dst = item
        if time.time() - t0 > args.budget:
            return 0
        src = os.path.join(tmp, name.replace("/", "__"))
        with zipfile.ZipFile(z) as zf, open(src, "wb") as f:
            f.write(zf.read(name))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", part], check=True)
        os.replace(part, dst)
        os.remove(src)
        return 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = sum(ex.map(one, todo))
    print(f"converted now: {done}, remaining: {len(todo) - done}", flush=True)


def wav_seconds(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return w.getnframes() / w.getframerate()


def build_csv(args):
    people = analysis_people(args.base).reset_index(drop=True)
    rows = []
    for key in people.interview__key:
        for p in sorted(glob(os.path.join(args.out, "audio", key, "*.wav"))):
            stem = os.path.basename(p)[:-4]
            rows.append({"uuid": f"{key}__{stem}", "filename": f"{key}/{stem}", "user": key,
                         "duration": round(wav_seconds(p), 3)})
    rec = pd.DataFrame(rows).merge(people, left_on="user", right_on="interview__key").drop(columns="interview__key")
    rec[["phq", "gad"]] = rec[["phq", "gad"]].astype(int)

    # split by person, stratified by version and PHQ > 9
    people["stratum"] = people.version.astype(str) + "_" + (people.phq > 9).astype(str)
    shuffled = people.sample(frac=1, random_state=0)
    split = {}
    for _, g in shuffled.groupby("stratum"):
        n = len(g); n_tr = round(0.70 * n); n_va = round(0.15 * n)
        for i, key in enumerate(g.interview__key):
            split[key] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    rec["split"] = rec.user.map(split)

    for s in ("train", "val", "test"):
        d = rec[rec.split == s].drop(columns="split").rename(columns={"duration": DURATION_COLUMN[s]})
        d.to_csv(os.path.join(args.out, f"{s}.csv"), index=False)
        print(f"{s}: {d.user.nunique()} people, {len(d)} recordings, "
              f"<30 s: {(d[DURATION_COLUMN[s]] < 30).sum()}, PHQ>9: {(d.drop_duplicates('user').phq > 9).mean():.1%}, "
              f"GAD>9: {(d.drop_duplicates('user').gad > 9).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audio", "csv"])
    ap.add_argument("--base", required=True, help="Bridge_baseline folder")
    ap.add_argument("--out", required=True, help="output folder (e.g. Bridge_baseline/data/clean/dam1)")
    ap.add_argument("--budget", type=float, default=1e9, help="seconds; stop converting after this")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    convert(a) if a.step == "audio" else build_csv(a)
