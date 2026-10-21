"""Generate synthetic data that satisfies the kirad data contract (https://pipotheavenger.github.io/kintsugi-rad-docs/data-contract/).

Writes, per mode, <out-dir>/<mode>/{train,val,test}.csv and 16 kHz mono int16 WAVs under
<out-dir>/<mode>/audio/<split>/<uuid>.wav. Only needs numpy (stdlib `wave` writes the WAVs).

Modes
  random    labels are drawn independently of the audio  -> negative control (no signal)
  learnable labels are encoded in the audio spectrum      -> the model can learn them
              phq (depression) -> fundamental frequency  110..333 Hz (log scale)
              gad (anxiety)    -> noise-band centre freq 600..3400 Hz (log scale)
            Frequency (not loudness) carries the signal because the dataset peak-normalises audio.

Stages of the test plan (always pass a different --out-dir per stage):
  1  --mode random   --n-train 8   --n-val 4   --n-test 4   --min-dur 60 --max-dur 61
  2  --mode random   --n-train 7   --n-val 4   --n-test 4   --min-dur 60 --max-dur 61 --edge
  3  --mode learnable --n-train 200 --n-val 40  --n-test 40      (+ control: --mode random, same sizes)
  4  --mode learnable --n-train 500 --n-val 100 --n-test 100

Durations default to 60-61 s: DAM 1 trains with `random_adjacent_pair`, which needs >= 2 windows
of 30 s. Labels are stratified so every split has samples on both sides of every metric cutoff
(phq > 9, > 14; gad > 4, > 9, > 14) whenever the split is large enough.
"""

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000  # kirad.constants.EXPECTED_SAMPLE_RATE
MIN_DURATION_FILTER = 30.0  # KintsugiMetadata default min_duration (inclusive)
# Score bins whose edges are the metric cutoffs (kirad.constants.METRIC_TARGET_CUTOFFS).
PHQ_BINS = [(0, 9), (10, 14), (15, 27)]
GAD_BINS = [(0, 4), (5, 9), (10, 14), (15, 21)]
EDGE_DURS = (10.0, 20.0, 29.0, 30.0)
SPLITS = ("train", "val", "test")
# Duration column name per split, as the DAM 1 config reads it (`duration_column`).
DURATION_COLUMN = {"train": "prompt_vad_trimmed_duration", "val": "prompt_vad_cut_duration",
                   "test": "prompt_vad_cut_duration"}


def stratified_scores(rng, n, bins):
    """n integer scores, cycling through the bins (shuffled) so every bin is covered."""
    order = rng.permutation(np.arange(n) % len(bins))
    return np.array([rng.integers(bins[k][0], bins[k][1] + 1) for k in order])


def envelope(rng, n, rate_hz=4.0):
    """Slow random amplitude envelope in [0.3, 1], mimicking syllable-like bursts."""
    knots = rng.uniform(0.3, 1.0, size=int(n / SR * rate_hz) + 2)
    x = np.linspace(0, len(knots) - 1, n)
    return np.interp(x, np.arange(len(knots)), knots).astype(np.float32)


def synth_audio(rng, n, phq_latent, gad_latent):
    """Voiced harmonic stack (f0 from phq) + band noise (centre from gad) + faint noise."""
    t = np.arange(n, dtype=np.float32) / SR
    f0 = 110.0 * 2 ** (1.6 * phq_latent / 27) * rng.uniform(0.99, 1.01)
    fc = 600.0 * 2 ** (2.5 * gad_latent / 21) * rng.uniform(0.98, 1.02)

    # Voiced part: 6 harmonics with slight vibrato, phase accumulated for a clean sweep.
    vib = 1 + 0.01 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0 * vib) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 7)).astype(np.float32)
    voiced *= envelope(rng, n)

    # Band noise: white noise filtered with a Gaussian window in log-frequency (0.25 oct).
    spec = np.fft.rfft(rng.standard_normal(n).astype(np.float32))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = 1.0
    spec *= np.exp(-0.5 * (np.log2(freqs / fc) / 0.25) ** 2)
    band = np.fft.irfft(spec, n).astype(np.float32)
    band = 0.6 * band / (band.std() + 1e-9) * envelope(rng, n)

    audio = voiced / (voiced.std() + 1e-9) + band + 0.02 * rng.standard_normal(n).astype(np.float32)
    audio *= 0.5 / (np.abs(audio).max() + 1e-9)  # the dataset re-normalises; keep headroom
    return audio


def write_wav(path, audio):
    """16 kHz, mono, int16 PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def build_split(rng, mode, split, n, min_dur, max_dur, out, extra_durs=()):
    """Write one split's audio and return its metadata rows."""
    n_total = n + len(extra_durs)
    phq = stratified_scores(rng, n_total, PHQ_BINS)
    gad = stratified_scores(rng, n_total, GAD_BINS)
    if mode == "learnable":
        phq_lat, gad_lat = phq, gad
    else:  # random: same marginals, but the audio latents are unrelated to the labels
        phq_lat = rng.permutation(stratified_scores(rng, n_total, PHQ_BINS))
        gad_lat = rng.permutation(stratified_scores(rng, n_total, GAD_BINS))

    durs = [float(d) for d in rng.uniform(min_dur, max_dur, size=n)] + list(extra_durs)
    rows = []
    for i in range(n_total):
        kind = "edge" if i >= n else "synth"
        uuid = f"{mode}-{split}-{kind}-{i:05d}"
        n_samples = int(round(durs[i] * SR))
        dur = n_samples / SR  # duration column == real length exactly (30 s -> 30.0)
        write_wav(out / "audio" / split / f"{uuid}.wav",
                  synth_audio(rng, n_samples, phq_lat[i], gad_lat[i]))
        rows.append({
            "uuid": uuid,
            "filename": f"{split}/{uuid}",  # -> <audio_root>/<filename>.wav
            "phq": int(phq[i]),
            "gad": int(gad[i]),
            DURATION_COLUMN[split]: dur,  # seconds; name read by `duration_column`
        })
    return rows


def write_csv(path, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_split(split, rows, out, min_dur):
    """Verify the on-disk contract and report coverage of the metric cutoffs."""
    assert len({r["uuid"] for r in rows}) == len(rows), "duplicate uuid"
    for r in rows:
        with wave.open(str(out / "audio" / f"{r['filename']}.wav")) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
            assert abs(w.getnframes() / SR - r[DURATION_COLUMN[split]]) < 1e-9
        assert 0 <= r["phq"] <= 27 and 0 <= r["gad"] <= 21
    kept = [r for r in rows if r[DURATION_COLUMN[split]] >= min_dur]
    phq = np.array([r["phq"] for r in rows])
    gad = np.array([r["gad"] for r in rows])
    cov = {f"phq>{c}": f"{(phq > c).sum()}/{len(rows)}" for c in (9, 14)}
    cov.update({f"gad>{c}": f"{(gad > c).sum()}/{len(rows)}" for c in (4, 9, 14)})
    print(f"  {split:5s} rows={len(rows):4d}  after min_duration={min_dur:g}s filter: {len(kept):4d}  {cov}")
    return len(kept)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("random", "learnable"), required=True)
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--min-dur", type=float, default=60.0, help="seconds (default 60)")
    p.add_argument("--max-dur", type=float, default=61.0, help="seconds (default 61)")
    p.add_argument("--edge", action="store_true",
                   help=f"add {EDGE_DURS} s recordings to train to test the duration filter")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty <out-dir>/<mode>")
    a = p.parse_args()

    if a.min_dur > a.max_dur:
        sys.exit("--min-dur must be <= --max-dur")
    out = a.out_dir.expanduser() / a.mode
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        sys.exit(f"{out} is not empty; use another --out-dir per stage (or --overwrite).")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    print(f"mode={a.mode} seed={a.seed} -> {out}")
    all_rows = {}
    for split in SPLITS:
        extra = EDGE_DURS if (a.edge and split == "train") else ()
        all_rows[split] = build_split(rng, a.mode, split, sizes[split], a.min_dur, a.max_dur, out, extra)
        write_csv(out / f"{split}.csv", all_rows[split])

    print("Contract check (files re-read from disk):")
    kept = {s: check_split(s, all_rows[s], out, MIN_DURATION_FILTER) for s in SPLITS}

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    (out / "manifest.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(a).items()}, "rows": {s: len(r) for s, r in all_rows.items()},
         "expected_after_filter": kept, "git_commit": commit}, indent=2))

    print(f"""
Config (per split section of data_params -> metadata=...KintsugiMetadata; drop merge_rows_by_subject):
  train:    df_or_dataset_version: {out}/train.csv   audio_root: {out}/audio   duration_column: prompt_vad_trimmed_duration
  val_test: df_or_dataset_version: {out}/<val|test>.csv   audio_root: {out}/audio   duration_column: prompt_vad_cut_duration
Expected rows in train after filter_metadata: {kept['train']}""")


if __name__ == "__main__":
    main()
