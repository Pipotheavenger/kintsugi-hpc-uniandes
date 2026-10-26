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

## 9. DAM 3, paso A2: `logmel_energies.npy`, el perfil espectral de referencia (2026-10-26)

### 9.1 Qué es

Antes de entrar a Whisper, cada ventana de 30 s se convierte en un **log-mel**: una matriz de **80 bandas de
frecuencia × 3000 instantes** (uno cada 10 ms). Cada celda es la energía, en escala logarítmica, de esa banda en ese
instante. Las bandas van de graves (banda 0) a agudos (banda 79), repartidas en escala mel (como las percibe el oído),
entre 0 y 8 kHz.

`logmel_energies.npy` es un vector de **80 números, uno por banda**: la energía promedio que "debería" tener cada
banda. Es un **perfil espectral de referencia**.

### 9.2 Qué hace el código con él

Por cada ventana y cada banda: calcula el promedio en el tiempo de esa banda, lo resta y suma el valor de referencia.

```text
features[banda, t] += ideal[banda] - promedio_t(features[banda, :])
```

Resultado: todas las ventanas de todas las grabaciones quedan con **el mismo perfil promedio**, y se conserva intacta
la variación en el tiempo, que es lo que aporta la voz.

| Modelo | Dónde se aplica | De dónde salen los 80 valores |
|---|---|---|
| DAM 1 (`config.yaml` actual) | Dentro del modelo, en GPU: `kirad/feature_extraction_whisper.py:132-134` | Constante `IDEAL_LOGMEL_ENERGIES`, `kirad/constants.py:149` |
| DAM 1 (`config_dam1.0.yaml`, versión vieja) | En el dataset | Archivo `logmel_energies.npy` (línea 100) |
| DAM 2 y DAM 3 | En el dataset: `kirad/datasets_and_dataloaders.py:749-753` (misma operación) | Archivo `logmel_energies.npy` (privado, no viene en el repo) |

Además, `scripts/launcher.py:224` hace `np.load()` de ese valor para registrarlo en W&B, así que en el YAML tiene que
ser **una ruta a un `.npy`**, no una lista de números.

### 9.3 Por qué es importante

Cada grabación trae el "color" del micrófono, el teléfono, la sala o la distancia (un celular corta graves, un
micrófono barato realza agudos, el eco engorda ciertas bandas). Ese color es constante durante la grabación, así que
vive en el promedio de cada banda. Reemplazar ese promedio por el perfil de referencia:

1. **Hace que el modelo aprenda de la voz y no del aparato.** Sin esto, el modelo podría usar el tipo de dispositivo
   como atajo (p. ej. si un grupo con PHQ alto se grabó con el mismo teléfono). Es el mismo riesgo que mezclar audio de
   8 y 16 kHz.
2. **Hace comparables grabaciones de distintas fuentes**, clave si BRIDGE-MH junta países o dispositivos.
3. **Deja la entrada en el rango que Whisper espera**: no la centra en cero, la lleva a un perfil parecido al de la
   voz con la que Whisper fue preentrenado.

### 9.4 Los valores que Kintsugi dejó fijos para DAM 1

Escala del log-mel de Whisper (`(log10 + 4) / 4`). Frecuencias aproximadas calculadas por nosotros con la escala mel
"slaney" de 80 bandas entre 0 y 8 kHz que usa el código (no vienen en el repo).

| Bandas | Frecuencia aprox. | Valores |
|---|---|---|
| 0–9 | 40–370 Hz | 0.349 0.586 0.791 0.928 **0.983 0.984** 0.960 0.939 0.937 0.932 |
| 10–19 | 400–750 Hz | 0.916 0.893 0.864 0.833 0.800 0.778 0.756 0.730 0.704 0.680 |
| 20–29 | 780–1,150 Hz | 0.658 0.635 0.614 0.592 0.572 0.552 0.537 0.522 0.510 0.499 |
| 30–39 | 1.2–1.9 kHz | 0.489 0.478 0.468 0.473 0.463 0.448 0.467 0.451 0.464 0.449 |
| 40–49 | 2–2.9 kHz | 0.455 0.437 0.432 0.422 0.412 0.401 0.393 0.388 0.384 0.385 |
| 50–59 | 3–4.3 kHz | 0.382 0.377 0.372 0.365 0.354 0.343 0.328 0.314 0.306 0.296 |
| 60–69 | 4.5–6.3 kHz | 0.293 0.286 0.275 0.269 0.258 0.248 0.241 0.234 0.225 0.209 |
| 70–79 | 6.5–7.9 kHz | 0.194 0.183 0.176 0.171 0.166 0.159 0.143 0.118 0.080 **0.037** |

Mín 0.0375 (banda 79), máx 0.9844 (banda 5), media 0.477. Forma: bajo en la banda 0 (casi no hay voz a 40 Hz), **pico
en las bandas 4–5 (~190–220 Hz, tono fundamental de la voz y primeros armónicos)** y caída gradual hacia 8 kHz. Es la
forma típica del espectro promedio de la voz humana.

### 9.5 Por qué está fijo en el código (hechos vs. inferencias)

**Hechos (en el código):**

- No hay ningún comentario que explique el origen de los valores. En la documentación lo anotamos como "perfil fijo
  de origen desconocido" (`docs/hardcoded.md` del repo de docs).
- La versión vieja de DAM 1 (`config_dam1.0.yaml`) leía el archivo desde el dataset, como DAM 2/3.
- La versión actual de DAM 1 calcula el log-mel dentro del modelo (`preprocessor: false`) con la constante, y
  `kirad/feature_extraction_whisper.py:78` dice: *"To enable model compilation, ideal_logmel_energies should be
  registered as a buffer"*. El launcher tiene un modo `compile` que exporta el modelo a `.pt2`.

**Inferencias (no confirmadas con Kintsugi):**

1. **Origen de los números:** muy probablemente es el promedio del log-mel por banda sobre muchas grabaciones de sus
   datos de entrenamiento (inglés, EE. UU., sus dispositivos). Es la forma estándar de construir este perfil y la
   curva tiene exactamente esa forma.
2. **Por qué lo fijaron:** para que el modelo exportado sea **autocontenido**: el `.pt2` lleva la normalización dentro y
   en producción no hay que cargar un archivo aparte ni arriesgarse a usar uno equivocado. De paso, el log-mel en GPU
   es más rápido.
3. Es casi seguro que la constante es el contenido del mismo `logmel_energies.npy` que antes leía el dataset.

### 9.6 Qué hicimos

- `hpc/configs/logmel_energies.npy`: los 80 valores de la constante, `float32`, forma `(80,)`, idénticos a
  `IDEAL_LOGMEL_ENERGIES` (`np.array_equal` = True). Así DAM 2/3 normalizan **exactamente igual que DAM 1**, que ya corrió.
- `hpc/make_logmel_energies.py`: lo regenera desde la constante (lee `kirad/constants.py` con `ast`, solo necesita
  numpy). `python hpc/make_logmel_energies.py` produce un archivo idéntico al del commit.

**Uso en los configs:** ruta **absoluta**
`/hpcfs/home/ing_quimica/sf.valencia/Kintusgi/kintsugi-hpc-uniandes/hpc/configs/logmel_energies.npy`. Una ruta
relativa se resuelve contra `--experiment-dir` (la carpeta de `research/`), no contra `hpc/configs/`.

### 9.7 Pendiente para datos reales

El perfil refleja el audio de Kintsugi. Para BRIDGE-MH (otro idioma, otros dispositivos, otros países) lo correcto
sería **recalcularlo con el promedio por banda de nuestros datos reales** (mismo log-mel que usa el código) o, al
menos, preguntar a Kintsugi cómo lo calcularon. Para las pruebas con datos sintéticos, la constante es suficiente.

## 10. DAM 3, pasos A3 y A4: generador con texto y los cuatro configs (2026-10-26)

### A3. `make_synth_data.py --with-text`

DAM 2 (y por tanto la cadena de DAM 3) lee una transcripción por audio en `<text_root>/<filename>.json`
(solo el campo `"text"`, `kirad/utils.py:read_whisper_timestamped_json`) y exige la columna `asr_output_id`
(su valor no se usa). Nuestros audios sintéticos no tienen voz, así que el ASR no sirve: el generador escribe el texto.

- Nueva opción `--with-text`: escribe `<out>/<modo>/text/<split>/<uuid>.json` = `{"text": "..."}` y agrega
  `asr_output_id` (= `uuid`) a los CSV. Sin la opción, la salida es la misma de antes.
- Texto en inglés (BERT es `bert-base-uncased`): 32 palabras de relleno + 4 palabras del rango de PHQ
  (bajo / medio / alto) + 4 del rango de GAD (4 rangos), mezcladas.
  - `learnable`: los rangos salen de las etiquetas → BERT tiene señal.
  - `random`: salen de los mismos latentes aleatorios del audio → sin relación con las etiquetas.
- El chequeo del contrato ahora también verifica que exista cada JSON con texto no vacío y la columna `asr_output_id`.
- Probado: 8/4/4 en `learnable` y `random`; columnas `uuid,filename,phq,gad,<duración>,asr_output_id`.

### A4. Configs en `hpc/configs/`

Todos apuntan a `~/Kintusgi/Data/kintsugi_synth/dam3_stage1/random/` y a `logmel_energies.npy` con ruta absoluta,
sin dataset privado (`v2.x`, `data_sources`, `dataset_fold`), con `effective_batch_size` 2, `max_epochs` 3,
`training_examples_per_eval` 4 y `cpus_per_worker` 4. Modelo, LoRA, pérdidas y optimizador: iguales al original.
Se validó estáticamente que todas las claves existen en las clases de `kirad` (los modelos usan `extra="forbid"`).

| Config | Base | Cambios propios | Pendiente |
|---|---|---|---|
| `dam2.yaml` (B1) | `jun_2024/config.yaml` | `text_dir` eliminado (no es campo válido); `text_root` en la metadata | — |
| `dam2_cache.yaml` (B2) | `dam2.yaml` | ventanas `all` + `pad_last_chunk_to_full` en todos los splits | `ckpt_path` = `PENDIENTE_URL_DAM2_B1` |
| `dam3_phase1.yaml` (B3) | `aug_2024/config_llma.yaml` | `audio_dir_or_feature_store_bucket` eliminado (no es campo válido); `audio_root` en la metadata; se mantiene `max_duration: 90` en train | `backbone_cache` = `PENDIENTE_URL_BACKBONE_CACHE_B2` |
| `dam3_phase2.yaml` (B4) | `aug_2024/config.yaml` | `ckpt_path` sigue `""` como upstream (Whisper de audio preentrenado) | `llma_ckpt` = `PENDIENTE_URL_LLMA_B3` |

Los marcadores `PENDIENTE_...` hacen fallar el job de forma explícita si se lanza antes de llenarlos.

**Decisión importante (B2 ↔ B3).** `broadcast_collate` (`kirad/dataset_utils.py:675-715`) exige que, por cada
grabación, el número de vectores del caché sea compatible con el número de ventanas de audio. La fase 1 usa ventanas
`all` + `pad_last_chunk_to_full` (techo de duración/30). Por eso el caché de DAM 2 se genera con esa misma
configuración de ventanas en los tres splits (en entrenamiento DAM 2 usa `single_random`, que daría 1 vector por audio
y rompería la fase 1).

**`.gitignore`.** El `.gitignore` original tiene la regla `*cache*`, que ignoraba `dam2_cache.yaml`; se agregó la
excepción `!hpc/configs/dam2_cache.yaml`.

**Duración de los audios para esta etapa:** exactamente 60 s (`--min-dur 60 --max-dur 60`), para que el relleno de la
última ventana no genere ventanas casi vacías (con 60.5 s habría una tercera ventana con 0.5 s de audio).

**Comando de datos (Hypatia, desde `~/Kintusgi`):**

```bash
python kintsugi-hpc-uniandes/hpc/make_synth_data.py --mode random --n-train 8 --n-val 4 --n-test 4 \
    --min-dur 60 --max-dur 60 --with-text --out-dir Data/kintsugi_synth/dam3_stage1
```

**Datos de la etapa (2026-10-26).** Hypatia actualizada a `afa226f`; generados con el comando anterior en
`~/Kintusgi/Data/kintsugi_synth/dam3_stage1/random/` (audio + `text/`): train 8, val 4, test 4, todos de 60 s,
0 descartados por el filtro de 30 s, todos los cortes de PHQ/GAD cubiertos en cada split. Chequeo del contrato OK.

## 11. DAM 3, paso B1: DAM 2 entrenado (2026-10-26)

**Job `784410`**, run W&B **`r2zzy01k`** (`dam3_stage1_B1_dam2`), commit `afa226f`.

```text
sacct → COMPLETED  ExitCode 0:0  Elapsed 00:03:24  MaxRSS 14,810,832K (~14.8 GB de RAM del nodo)
```

- Datos: train 8/8, val 4/4, test 4/4 (0 descartados). El tokenizer y el modelo BERT cargaron con transformers 5.20.1.
- 3 épocas × 4 batches; validaciones durante el entrenamiento; test completo.
- Test (sin significado, datos `random` y 4 audios): `metrics/test/auroc@0%i` 0.610, depresión 0.583,
  ansiedad 0.639, `metrics/test/total_loss` 33.21.
- Aviso nuevo, no bloquea: *"Skipping evaluation of threshold method 'coral' because its thresholds [...] are not
  sorted"*: con 3 épocas los umbrales CORAL aprendidos siguen casi en 0 y desordenados, así que ese método de umbral
  se salta; los demás se evalúan.
- Costo de B1 con 16 audios: 3 min 24 s, ~14.8 GB de RAM (el sbatch pide 32 GB).

**URL del modelo para B2 (en `dam2_cache.yaml`, `ckpt_path`):**
`https://forge.coreweave.com/sebastian-valencia-world-bank-/kintsugi-hpc-uniandes/artifacts/model/model-r2zzy01k/best`

**Formato de URLs de W&B (importante).** La interfaz muestra `https://forge.coreweave.com/wandb/<entity>/...`, con un
segmento `/wandb/` extra. `kirad/utils.py:get_wandb_run_name_tag_from_artifact_url` parte la URL por `/` y exige
6–7 partes: `<host>/<entity>/<project>/artifacts/<tipo>/<nombre>/<tag>`. Con `/wandb/` serían 8 y falla. Hay que
escribir las URLs **sin** `/wandb/`; el host se ignora, solo cuentan las posiciones. El artefacto del modelo se llama
`model-<id de la corrida>` (lo crea `WandbLogger` con `log_model=True`), con alias `best`.
