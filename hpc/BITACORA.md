# Bitácora — kintsugi-hpc-uniandes

Registro de cada paso de la puesta en marcha en Hypatia: qué se hizo, por qué, comandos y
resultado. Lo más reciente va al final.

## 1. Datos sintéticos — etapa 1 (humo)

**Objetivo.** Generar el conjunto mínimo para probar que el pipeline corre de punta a punta
(entrenamiento, al menos una validación y evaluación final en test). No se espera que el
modelo aprenda.

**Generador.** `hpc/make_synth_data.py` (solo numpy). Escribe `train.csv`, `val.csv`,
`test.csv` y WAV mono de 16 kHz que cumplen el contrato de datos del código:
columnas `uuid, filename, phq, gad` + una de duración según el split
(`prompt_vad_trimmed_duration` en train, `prompt_vad_cut_duration` en val/test, como las lee el
config de DAM 1); audios en `<out>/random/audio/<split>/<uuid>.wav`.

> Cambio posterior (2026-10-21): la primera versión escribía además `asr_output_id`, `user` y las
> dos columnas de duración en todos los splits. Se quitaron porque DAM 1 no las usa
> (`asr_output_id`: solo texto; `user`: solo `merge_rows_by_subject`). Hay que regenerar la etapa 1.
Duración 60–61 s porque DAM 1 entrena con `random_adjacent_pair` (2 ventanas de 30 s).
Etiquetas estratificadas para que cada split tenga casos a ambos lados de los cortes de las
métricas (phq > 9, > 14; gad > 4, > 9, > 14).

**Comando (Hypatia, nodo de login).**

```bash
conda activate kintsugi
cd ~/Kintusgi
python kintsugi-hpc-uniandes/hpc/make_synth_data.py --mode random --n-train 8 --n-val 4 --n-test 4 \
    --min-dur 60 --max-dur 61 --out-dir Data/kintsugi_synth/stage1
```

| | train | val | test | total |
|---|---|---|---|---|
| Audios | 8 | 4 | 4 | 16 |
| Esperados tras `filter_metadata` | 8 | 4 | 4 | |

**Resultado (2026-10-21).** OK. Corrió en el nodo de login (16 audios, segundos; no hizo
falta `srun`).

```text
mode=random seed=0 -> Data/kintsugi_synth/stage1/random
  train rows=   8  after min_duration=30s filter:    8  {'phq>9': '5/8', 'phq>14': '2/8', 'gad>4': '6/8', 'gad>9': '4/8', 'gad>14': '2/8'}
  val   rows=   4  after min_duration=30s filter:    4  {'phq>9': '2/4', 'phq>14': '1/4', 'gad>4': '3/4', 'gad>9': '2/4', 'gad>14': '1/4'}
  test  rows=   4  after min_duration=30s filter:    4  {'phq>9': '2/4', 'phq>14': '1/4', 'gad>4': '3/4', 'gad>9': '2/4', 'gad>14': '1/4'}
Expected rows in train after filter_metadata: 8
```

Ubicación en Hypatia: `/hpcfs/home/ing_quimica/sf.valencia/Kintusgi/Data/kintsugi_synth/stage1/random/`
(`train.csv`, `val.csv`, `test.csv`, `audio/`, `manifest.json`). En el config usar esta ruta
absoluta, no la relativa que imprime el script.

**Pendientes detectados.** El repo en Hypatia está en `~/Kintusgi/kintsugi-hpc-uniandes`, no en
`~/kintsugi-hpc-uniandes`: corregir `REPO_DIR` en `hypatia_sync.sh` y el `cd` de
`run_job.sbatch`. El entorno se activa con `conda activate kintsugi` sin `module load anaconda`.

## 2. W&B — cuenta, login en Hypatia y entity/project en el launcher

**Cuenta.** Creada en W&B (interfaz "Forge" de CoreWeave). Usuario `sebastian-valencia`,
entity (equipo) `sebastian-valencia-world-bank-` (el guion final es parte del nombre).
Plan: Academic trial (completar la solicitud para el plan académico gratuito).

**Login en Hypatia (2026-10-21).** La API key se guardó en `~/Kintusgi/key.txt` (fuera del repo) y:

```bash
wandb login "$(cat ~/Kintusgi/key.txt)"
# -> Appending key for api.forge.coreweave.com to your netrc file: ~/.netrc
# -> Currently logged in as: sebastian-valencia (sebastian-valencia-world-bank-)
```

**Cambio en el código.** `scripts/launcher.py`, `_setup_launcher` (líneas 138–139): entity y
project estaban en `None` y el launcher lanzaba `ValueError`. Se fijaron directamente en el código:

```python
wandb_entity = "sebastian-valencia-world-bank-"
wandb_project = "kintsugi-hpc-uniandes"
```

Las validaciones (`raise ValueError` si faltan) se mantienen. Pendiente para después: moverlos al
YAML del experimento.

## 3. Parche de `kipy` en el launcher

**Problema.** `scripts/launcher.py`, `_log_to_wandb` (línea 206) recorre `["kipy", "kirad"]` y llama
`get_package_info` para registrar versión y commit de cada paquete en W&B. `kipy` es un paquete
privado de Kintsugi que no está disponible, así que la corrida falla al arrancar.

**Cambio (2026-10-21).** Solo esa línea:

```python
# antes
for package in ["kipy", "kirad"]:
# después
for package in ["kirad"]:
```

**Nota.** `get_package_info("kirad")` necesita encontrar `kirad`: instalado con `pip install -e .`
en el entorno `kintsugi`, o con la variable `KINTSUGI_RAD_PATH`. Si no, falla en ese punto.

## 4. Whisper en caché (Hugging Face)

**Objetivo.** Descargar antes de la primera corrida el modelo que usa DAM 1
(`openai/whisper-small.en`) para que el job no dependa de bajarlo.

**Comando (2026-10-21), en un nodo de cómputo.**

```bash
srun --mem=8G --time=00:30:00 --pty bash      # asignó nodea-6
conda activate kintsugi
python -c "from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-small.en'); \
WhisperFeatureExtractor.from_pretrained('openai/whisper-small.en'); print('OK')"
ls ~/.cache/huggingface/hub
```

**Resultado.** OK. Se descargaron `config.json`, `model.safetensors` (967 MB) y
`preprocessor_config.json`. Caché en `~/.cache/huggingface/hub/models--openai--whisper-small.en`
(home compartido entre nodos).

**Hallazgo.** El nodo de cómputo (`nodea-6`) tiene salida a internet: la descarga desde Hugging
Face corrió a ~42 MB/s. El aviso "unauthenticated requests to the HF Hub" es solo un límite de
velocidad; no hace falta `HF_TOKEN` para este modelo público.

## 5. `run_job.sbatch` ajustado a Hypatia

**Cambios (2026-10-21).**

- Ruta del repo: `cd "$HOME/kintsugi-hpc-uniandes"` → `REPO_DIR="$HOME/Kintusgi/kintsugi-hpc-uniandes"`
  y `cd "$REPO_DIR"` (es donde está clonado).
- `export KINTSUGI_RAD_PATH="$REPO_DIR"`: `kirad.utils.get_package_path("kirad")` usa esta
  variable para encontrar el repo (setup.cfg y git) al registrar la versión en W&B.
- Entorno: se mantiene `module load anaconda`; la activación pasa de `source activate` a
  `eval "$(conda shell.bash hook)"` + `conda activate kintsugi`, dentro de `set +u ... set -u`
  porque los scripts de activación de conda fallan con variables no definidas bajo `set -u`.

## 6. Todo listo para la primera corrida (etapa 1)

Cambios hechos para que la etapa 1 corra sin tocar nada más (2026-10-21):

| Archivo | Cambio | Por qué |
|---|---|---|
| `hpc/configs/stage1.yaml` (nuevo) | Copia de `research/stable/ordinal_regression_mtl_mar_2024/config.yaml` con: `experiment_name: stage1_smoke_synth_random`; `effective_batch_size` 128→2; `max_epochs` 32→3; `training_examples_per_eval` 16000→4; `cpus_per_worker: 4`; `df_or_dataset_version` y `audio_root` absolutos a `~/Kintusgi/Data/kintsugi_synth/stage1/random/` (train, val y test por separado); se quitaron `merge_rows_by_subject`, `"v2.11.4"`, `data_sources` y `dataset_fold`. El resto (modelo, LoRA, pérdidas, optimizador, ventanas) igual. | Datos sintéticos de la etapa 1; con 8 audios de train y batch 2 hay 4 pasos por época y una validación cada 4 ejemplos (2 por época). `cpus_per_worker`: por defecto el launcher usa todas las CPUs del nodo como workers, pero el job solo pide 8. |
| `hpc/run_job.sbatch` | `export PYTHONPATH="$REPO_DIR"` | `scripts/launcher.py` hace `from kirad ...`; sin instalar el paquete Python no lo encuentra. |
| `hpc/run_job.sbatch` | Se quitaron `DATA_DIR`, `RUNS_DIR` y `mkdir -p $RUNS_DIR` | No los usa el código. |
| `hpc/run_job.sbatch` | Comentario de uso con el comando de la etapa 1 | |
| `hpc/hypatia_sync.sh` | `REPO_DIR` → `$HOME/Kintusgi/kintsugi-hpc-uniandes` | Ubicación real del clon. |
| `.gitignore` | `logs/`, `chkpts/` | El launcher escribe ahí (W&B en `./logs`, checkpoints en `./chkpts`) dentro del repo. |

**`--experiment-dir`.** El config usa `model_path: "model.DepAnxClassifier"`, que se importa desde
la carpeta del experimento. Como el config vive en `hpc/configs/`, se pasa
`--experiment-dir research/stable/ordinal_regression_mtl_mar_2024` para que encuentre `model.py`.

**Comando (desde `~/Kintusgi/kintsugi-hpc-uniandes`, rama `develop`).**

```bash
sbatch hpc/run_job.sbatch scripts/launcher.py train hpc/configs/stage1.yaml \
    --experiment-dir research/stable/ordinal_regression_mtl_mar_2024
```

**Verificaciones previas (Hypatia).**

- `sinfo -s`: la cola `gpu` existe (`nodei-gpu-[1-2]`, 15 días). También hay `gpu_economia` y `express`.
- `scontrol show node nodei-gpu-1 | grep -i gres` → `Gres=gpu:nvidia:3`: `--gres=gpu:1` es válido.
- `module avail anaconda` → `anaconda/conda4.12.0`, `anaconda/python3.9 (D)`.
- Prueba de activación igual a la del sbatch (`module purge; module load anaconda; eval "$(conda shell.bash hook)";
  conda activate kintsugi`) → `~/.conda/envs/kintsugi/bin/python`, `import torch, transformers` OK.

**Resultado (2026-10-21): etapa 1 OK.** Job `781898`, nodo `nodei-gpu-2`, commit `a1bf1bd`.

```text
sacct -j 781898 → COMPLETED  ExitCode 0:0  Elapsed 00:02:12
```

- GPU usada (`GPU available: True (cuda), used: True`), AMP 16 bits, 1 proceso DDP (NCCL).
- `filter_metadata`: train 8/8, val 4/4, test 4/4; 0 audios descartados por < 30 s.
- Modelo: 95.6 M parámetros, 7.6 M entrenables (LoRA + cabeza), 88.1 M congelados.
- Entrenamiento: 3 épocas × 4 batches (`Trainer.fit stopped: max_epochs=3 reached`). Hubo
  validaciones: se guardó checkpoint en W&B (artefacto `model-0w24804i`, run `0w24804i`).
- Test: corrió y reportó todas las métricas sin errores (p. ej. `metrics/test/auroc@0%i` 0.597,
  `metrics/test/total_loss` 33.09). Los valores no significan nada: datos `random` y 4 audios.

Criterio de la etapa 1 cumplido: corre de punta a punta, valida y evalúa en test.

**Avisos vistos (no bloquean).** `precision=16` está deprecado (equivale a `16-mixed`); Lightning
sugiere `srun` (innecesario con 1 GPU); `log_every_n_steps=5` > 4 batches por época, así que la
pérdida de train no se registra por paso en W&B; `find_unused_parameters=True` (fijo en el launcher)
cuesta algo de velocidad; atención "memory efficient" no determinista.

## 7. Revisión de DAM 3: qué pide y cuánto cuesta (2026-10-26)

Tarea acordada con Ignacio en la llamada del 22 de octubre. Fuente: `research/stable/ordinal_regression_mtl_aug_2024/`
(`model.py`, `config.yaml`, `config_llma.yaml`, `config_dam3.0.yaml`, `config_llma_dam3.0.yaml`) y la documentación
(`docs/experiments.md`, `docs/limitations.md` del repo de docs).

**Qué es.** En inferencia solo usa audio, pero para entrenarlo hace falta una cadena de modelos:

| Paso | Qué se entrena / corre | Clase / config | Necesita |
|---|---|---|---|
| 0 | Transcripciones ASR | `scripts/run_whisper_asr.py` (Whisper large-v3, `language="en"` fijo) | audio |
| 1 | DAM 2: Whisper + BERT, ambos con LoRA | `jun_2024`, `DepAnxClassifierWhisperBERT` | audio + transcripción (`<text_root>/<filename>.json`, campo `text`) |
| 2 | Caché de salidas de DAM 2 | `launcher.py cache` | checkpoint de DAM 2 (W&B) |
| 3 | Fase 1, `WhisperLLMA`: Whisper fine-tune completo para imitar (MSE) la mitad BERT `[768:1536]` del caché | `config_llma.yaml` | caché del paso 2 (`backbone_cache`, W&B URL) |
| 4 | Fase 2, `DepAnxClassifier`: Whisper y WhisperLLMA congelados, se entrena solo la cabeza | `config.yaml` | `llma_ckpt` del paso 3 (W&B URL) |

**Impacto en el contrato de datos (además de lo de DAM 1).**

- Para entrenar (pasos 0–3) hacen falta **transcripciones**: columna `asr_output_id` y un JSON por audio en `text_root`.
  En inferencia no.
- Todo el texto está pensado en inglés: ASR con `language="en"`, `bert-base-uncased` y `whisper-small.en`. Con audio
  en español habría que cambiar esas tres piezas.
- `ideal_logmel_energies: "logmel_energies.npy"` es obligatorio en DAM 2/3 y no está en el repo: hay que calcularlo
  con nuestros datos o quitar la clave.
- Fase 1 usa `max_duration: 90` en train (memoria); fase 2 usa `pad_last_chunk_to_full: true` (rellena la última
  ventana). Mínimo de 30 s y 16 kHz, igual que DAM 1.

**Bloqueos en los configs (docs/limitations.md 5–9).** `audio_dir_or_feature_store_bucket` y `text_dir` no son
campos válidos (usar `audio_root` / `text_root`); falta `logmel_energies.npy`; `llma_ckpt` y `backbone_cache` son
placeholders `<wandb_url_...>` que se llenan con nuestras propias corridas; los textos leen `asr_output_id`.

**Costo en Hypatia.** No medido. Comparado con DAM 1 son 4 corridas en serie en vez de 1; la fase 1 entrena todo el
encoder de Whisper small (no solo LoRA), mucho más que los 7.6 M de DAM 1; el ASR con large-v3 es una pasada extra
sobre todo el audio. Todo usa Whisper small o BERT base, así que debería caber en 1 GPU. Propuesta: medir cada paso
con datos sintéticos pequeños, como en la etapa 1. Para DAM 2 hacen falta transcripciones sintéticas (los audios
sintéticos no tienen voz).

## 8. DAM 3, paso A1: hardware y entorno (2026-10-26)

**GPU de la cola `gpu`** (`srun -p gpu --gres=gpu:1 nvidia-smi`): **NVIDIA Quadro RTX 6000, 24 GB** (23,040 MiB),
arquitectura Turing. Driver 520.61.05, **CUDA 11.8** (máximo que soporta el driver).

Implicaciones:

- 24 GB alcanzan para Whisper small, BERT base y el fine-tune completo del encoder de Whisper small (fase 1) con
  `batch_size` 1–2.
- Turing soporta fp16 (lo que usa el launcher) pero no bf16 ni FlashAttention 2 (`flash-attn` del env no sirve en
  esta GPU; DAM 1/2/3 no lo necesitan).
- Cualquier paquete compilado para CUDA 12 no corre con este driver.

**Entorno:** `import whisper_timestamped` → `ModuleNotFoundError`. Solo hace falta para el paso B0 (ASR de datos
reales); la cadena con datos sintéticos no lo usa. Al instalarlo, revisar antes con `pip install --dry-run` que no
reemplace `torch` por una versión CUDA 12.

**Revisión de dependencias (2026-10-26).**

- `pip install whisper-timestamped==1.15.8 --dry-run`: solo instalaría `whisper-timestamped`; todas sus dependencias
  ya están (`openai-whisper 20250625`, `dtw-python`, `Cython`) y `torch` se queda en **2.7.1+cu118**. Seguro de instalar.
- `import peft, transformers` → OK, **transformers 5.20.1**. DAM 1 corrió con esta versión (rama Whisper); la rama
  BERT de DAM 2 todavía no se ha probado con transformers 5.

## 9. DAM 3, paso A2: `logmel_energies.npy` (2026-10-26)

**Qué es.** Un vector de 80 números, uno por banda del log-mel que ve Whisper. Es el "perfil espectral ideal": la
energía promedio que debería tener cada banda.

**Qué hace el código con él.** Por cada ventana de 30 s y cada banda, resta el promedio en el tiempo de esa banda y
suma el valor ideal: `features += ideal - features.mean(dim=tiempo)`. Así toda ventana queda con el mismo perfil
promedio; se quita la "coloración" del micrófono, el teléfono o la sala, y queda la variación en el tiempo (la voz).

- DAM 1 lo hace dentro del modelo, en GPU, con la constante `IDEAL_LOGMEL_ENERGIES` (`kirad/constants.py:149`,
  usada en `kirad/feature_extraction_whisper.py:132-134`).
- DAM 2 y 3 lo hacen en el dataset (`kirad/datasets_and_dataloaders.py:749-753`) con la misma operación, pero leen
  los valores de un archivo (`ideal_logmel_energies: "logmel_energies.npy"`), que no viene en el repo.
- Además, `scripts/launcher.py:224` hace `np.load()` de ese valor para subirlo a W&B: tiene que ser una ruta a un
  `.npy`, no una lista en el YAML.

**Cómo se generó.** Se leyó la constante del código fuente (con `ast`, sin importar `kirad`) y se guardó tal cual:
`hpc/configs/logmel_energies.npy`, forma `(80,)`, `float32`, idéntico a la constante (`np.array_equal` = True).
Valores: mín 0.0375 (banda 79), máx 0.9844 (banda 5), media 0.477; más energía en bandas bajas, cae en las altas.

**Por qué así.** Con la constante, DAM 2/3 normalizan exactamente igual que DAM 1, que ya corrió.
**No verificado:** que sea igual al `logmel_energies.npy` privado de Kintsugi. Preguntarles, o recalcularlo con
nuestros datos reales.

**Uso en los configs (para A4).** Poner la ruta **absoluta**
`/hpcfs/home/ing_quimica/sf.valencia/Kintusgi/kintsugi-hpc-uniandes/hpc/configs/logmel_energies.npy`: una ruta
relativa se resuelve contra `--experiment-dir` (la carpeta de `research/`), no contra `hpc/configs/`.
