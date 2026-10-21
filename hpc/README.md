# hpc/ — running kintsugi-rad on Hypatia (Uniandes HPC)

Test fork of [Kintsugi-Labs/kintsugi-rad](https://github.com/Kintsugi-Labs/kintsugi-rad) for
running experiments on Hypatia. The code documentation lives in a separate, unchanged repo:
[kintsugi-rad-docs](https://pipotheavenger.github.io/kintsugi-rad-docs/).

| File | Runs on | What it does |
|---|---|---|
| `hypatia_sync.sh` | Hypatia | First time: clones `main`. Then: `git pull --ff-only` so Hypatia mirrors GitHub `main` exactly. `--env` creates/updates the `kintsugi` conda env (from `envs/kipy-gpu-full.yaml`) and runs `pip install -e .`. |
| `run_job.sbatch` | Hypatia | Slurm template (queue `gpu`). Loads anaconda, activates the env, logs the commit that ran, then `python "$@"`. |
| `make_synth_data.py` | anywhere (numpy only) | Synthetic CSV + 16 kHz WAVs that satisfy the [data contract](https://pipotheavenger.github.io/kintsugi-rad-docs/data-contract/), to test the pipeline end to end. |

## Workflow

```bash
# Mac: work on develop, ship by merging into main
git checkout main && git merge develop && git push origin main && git checkout develop

# Hypatia
ssh sf.valencia@hypatia.uniandes.edu.co
bash ~/kintsugi-hpc-uniandes/hpc/hypatia_sync.sh          # add --env the first time
```

Data and outputs live outside the repo (`~/data`, `~/runs`); never edit code on Hypatia.

## Pipeline test plan (synthetic data)

Move to the next stage only when the current one meets its criterion. Use a different
`--out-dir` per stage (the script refuses to write into a non-empty folder).

| Stage | Command | Audios | Success |
|---|---|---|---|
| 1 Smoke | `--mode random --n-train 8 --n-val 4 --n-test 4` | 16 | runs end to end: training, a validation, final test eval |
| 2 Filter | `--mode random --n-train 7 --n-val 4 --n-test 4 --edge` | 19 | 8 train rows after `filter_metadata` ("Remaining number of files" in the job log) |
| 3 Learning | `--mode learnable --n-train 200 --n-val 40 --n-test 40` | 280 | train and val loss go down |
| 3 Control | `--mode random --n-train 200 --n-val 40 --n-test 40` | 280 | val does **not** improve (if it does: data leak) |
| 4 Scale (opt.) | `--mode learnable --n-train 500 --n-val 100 --n-test 100` | 700 | clearer val gain; time per epoch |

```bash
python hpc/make_synth_data.py --mode random --n-train 8 --n-val 4 --n-test 4 --out-dir ~/data/synth/stage1
```

Durations default to 60–61 s (DAM 1 trains with `random_adjacent_pair`, which needs 2 windows of 30 s).
The script prints the paths to put in the config (`df_or_dataset_version`, `audio_root`,
`duration_column`); remove `merge_rows_by_subject`. Epochs: `training_params.max_epochs`.

| Stage | `effective_batch_size` | `training_examples_per_eval` | `max_epochs` |
|---|---|---|---|
| 1–2 | 2 | 4 | 2–3 |
| 3 | 4 | 100 | 8–10 |
| 4 | 8 | 250 | 8–10 |
