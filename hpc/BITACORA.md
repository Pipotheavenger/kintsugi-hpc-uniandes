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

## 12. DAM 3, paso B2: caché de DAM 2 (2026-10-26)

**Job `784413`**, run W&B **`0lxldli5`** (`dam3_stage1_B2_dam2_cache`), commit `9cb047a`, modo `cache`.

```text
sacct → COMPLETED  ExitCode 0:0  Elapsed 00:01:16  MaxRSS 2,433,344K (~2.4 GB)
```

- Descargó el modelo de B1 desde W&B: `Downloading large artifact 'model-r2zzy01k:best', 804.15MB`. Primera vez que el
  código usa una URL de W&B: funcionó con el formato sin `/wandb/`.
- Cargó los pesos completos: no aparece "Performing partial initialization" ni errores en el log.
- Pasó los 16 audios por DAM 2 (train 4 batches, val 2, test 2): `Global cache size: 16`.
- Subió el artefacto `backbone_cache-0lxldli5` (tag `v0`).
- Costo de B2 con 16 audios: 1 min 16 s, ~2.4 GB de RAM (la mayor parte, cargar el modelo).

**URL del caché para B3 (en `dam3_phase1.yaml`, `backbone_cache`):**
`https://forge.coreweave.com/sebastian-valencia-world-bank-/kintsugi-hpc-uniandes/artifacts/backbone_cache/backbone_cache-0lxldli5/v0`

## 13. DAM 3, paso B3: fase 1 (`WhisperLLMA`) (2026-10-26)

Se lanzó dos veces (mismo config): job **`784414`** (09:06, 2 min 17 s, ~15.0 GB RAM) y job **`784453`**
(10:24, 2 min 28 s, ~16.5 GB RAM), ambos `COMPLETED 0:0`. Se usa la **segunda**: run W&B **`6xlyjmro`**.

- Descargó el caché de B2 y `Filtered out 0 files that are not in the backbone cache`.
- No hubo error de `broadcast_collate`: el número de vectores del caché coincide con las ventanas de audio (gracias a
  generar el caché con `all` + `pad_last_chunk_to_full`, paso 10).
- 3 épocas, validaciones, test completo: `metrics/test/total_loss` (MSE contra la mitad BERT del caché) = **0.136**.
- Artefacto `model-6xlyjmro` (alias `best`). El aviso *"already exists with the same content"* aparece porque el
  mejor checkpoint y el último son el mismo archivo.

**Carpeta de checkpoints compartida.** Como ambas corridas tienen el mismo `experiment_name`, las dos escribieron en
`chkpts/dam3_stage1_B3_phase1_llma/` (Lightning avisa *"exists and is not empty"* y agrega `-v1` a los archivos de la
segunda). No afecta lo que se usa después: cada corrida sube a W&B sus propios checkpoints, y la fase 2 lee el de
`6xlyjmro` desde W&B. Cada checkpoint pesa ~1 GB (encoder completo de Whisper), conviene limpiar la carpeta de vez en
cuando. Para no mezclar corridas, cambiar `experiment_name` en cada repetición.

**Detalle cosmético del launcher.** Los checkpoints se llaman `epoch-2_metric-0.000.ckpt`: en `scripts/launcher.py`
el nombre se arma con `filename="epoch-{epoch}_metric-{monitor_loss_name:.3f}"`, que no es un f-string, así que
Lightning busca una métrica llamada literalmente `monitor_loss_name`, no la encuentra y pone 0.000. No afecta el
entrenamiento ni qué checkpoint es el mejor.

**Costo de B3 con 16 audios:** ~2.5 min, ~16.5 GB de RAM.

**URL para B4 (en `dam3_phase2.yaml`, `llma_ckpt`):**
`https://forge.coreweave.com/sebastian-valencia-world-bank-/kintsugi-hpc-uniandes/artifacts/model/model-6xlyjmro/best`

## 14. DAM 3, paso B4: fase 2 y cadena completa (2026-10-26)

**Job `784460`**, run W&B **`lg6r142h`** (`dam3_stage1_B4_phase2`), commit `55cfa7a`.

```text
sacct → COMPLETED  ExitCode 0:0  Elapsed 00:02:28  MaxRSS 3,440,784K (~3.4 GB)
```

- Cargó el modelo de la fase 1 (`model-6xlyjmro:best`) con `strict=True` sin error (la carga ocurre al construir
  el modelo; si un peso no coincidiera, el job habría fallado antes de entrenar).
- Datos 8/4/4; 3 épocas; validaciones; test completo.
- Sin los avisos de `find_unused_parameters` ni de atención no determinista de B1/B3: coherente con que los dos
  Whisper están congelados y solo se entrena la cabeza.
- Test (sin significado: datos `random`, 4 audios por split): `metrics/test/auroc@0%i` 0.653, depresión 0.792,
  ansiedad 0.556, `total_loss` 33.07.

### Resultado: la cadena de DAM 3 corre de punta a punta en Hypatia

| Paso | Qué hace | Job | Run W&B → artefacto | Tiempo | RAM (MaxRSS) |
|---|---|---|---|---|---|
| B1 | Entrena DAM 2 (Whisper + BERT, LoRA) | 784410 | `r2zzy01k` → `model-r2zzy01k:best` | 3 min 24 s | ~14.8 GB |
| B2 | Caché de DAM 2 (audio + BERT por ventana) | 784413 | `0lxldli5` → `backbone_cache-0lxldli5:v0` | 1 min 16 s | ~2.4 GB |
| B3 | Fase 1: `WhisperLLMA` (Whisper completo, MSE a la mitad BERT) | 784453 | `6xlyjmro` → `model-6xlyjmro:best` | 2 min 28 s | ~16.5 GB |
| B4 | Fase 2: cabeza PHQ/GAD sobre Whisper + WhisperLLMA congelados | 784460 | `lg6r142h` | 2 min 28 s | ~3.4 GB |
| | **Total cadena** | | | **~9.5 min** | pico ~16.5 GB |

GPU: Quadro RTX 6000 (24 GB), 1 GPU por paso. Con 16 audios los tiempos miden sobre todo el costo fijo (cargar
modelos, descargar artefactos de W&B), no el de entrenar con datos reales.

**Lo que hizo falta para que corriera** (pasos 8–13): `whisper-timestamped` instalado; `logmel_energies.npy` desde la
constante de DAM 1; transcripciones sintéticas (`--with-text`); cuatro configs con las claves inválidas de upstream
corregidas; caché generado con las mismas ventanas que la fase 1; URLs de W&B sin el segmento `/wandb/`; excepción
en `.gitignore` para `dam2_cache.yaml`.

**Siguiente:** prueba con señal (~280 audios `learnable` con texto) y control (`random`, mismo tamaño), para ver si
la cadena aprende y para medir el costo real por paso.

## 15. DAM 3, etapa 3: prueba con señal (`learnable`) y control (`random`) — preparación (2026-10-26)

**Objetivo.** Ver si la cadena de DAM 3 aprende cuando hay relación entre audio/texto y etiqueta, y que no "aprenda"
cuando no la hay (control contra fugas). De paso, medir el costo con más datos que en la etapa 1.

**Datos.** 200 / 40 / 40 audios de 60 s exactos, con texto, en dos carpetas separadas:

```bash
# Hypatia, en un nodo de cómputo (son ~280 audios y ~0.5 GB por conjunto)
srun --mem=8G --time=00:30:00 --pty bash
conda activate kintsugi && cd ~/Kintusgi
python kintsugi-hpc-uniandes/hpc/make_synth_data.py --mode learnable --n-train 200 --n-val 40 --n-test 40 \
    --min-dur 60 --max-dur 60 --with-text --out-dir Data/kintsugi_synth/dam3_stage3_learnable
python kintsugi-hpc-uniandes/hpc/make_synth_data.py --mode random --n-train 200 --n-val 40 --n-test 40 \
    --min-dur 60 --max-dur 60 --with-text --out-dir Data/kintsugi_synth/dam3_stage3_random
exit
```

En `learnable`, la etiqueta está en el audio (tono ↔ PHQ, banda de ruido ↔ GAD) y en el texto (palabras por rango);
en `random`, ni el audio ni el texto tienen relación con la etiqueta.

**Configs.** `hpc/configs/dam3_stage3_learnable/` y `hpc/configs/dam3_stage3_random/`, cada una con los cuatro
configs de la cadena. Son copias de los de la etapa 1 con: rutas de datos de su conjunto; `experiment_name` propio
(`dam3_stage3_<modo>_B1_dam2`, …), así cada corrida tiene su carpeta de checkpoints; parámetros de la etapa 3 del
plan: `effective_batch_size` 4, `training_examples_per_eval` 100 (2 validaciones por época), `max_epochs` 8; y las
URLs de W&B reiniciadas a `PENDIENTE_...` (no deben heredarse de la etapa 1).

**`.gitignore`.** La excepción pasa a `!hpc/configs/**/dam2_cache.yaml` para cubrir las subcarpetas.

**Criterio de éxito.**

| | `learnable` | `random` (control) |
|---|---|---|
| B1 DAM 2 | pérdida baja en train y val; AUROC de test claramente > 0.5 | val no mejora; AUROC ~0.5 |
| B3 fase 1 | MSE de val baja | MSE baja (imita a BERT igual), pero sin información de la etiqueta |
| B4 fase 2 | val mejora; AUROC de test claramente > 0.5 | val no mejora; AUROC ~0.5 (si mejora: fuga) |

**Datos generados (2026-10-26)** en un `srun` (`nodea-6`): `dam3_stage3_learnable/learnable/` y
`dam3_stage3_random/random/`, 200/40/40 cada uno, 60 s, con texto, 0 descartados, misma distribución de etiquetas en
ambos (p. ej. train: phq>9 133/200, phq>14 66/200, gad>4 150/200, gad>9 100/200, gad>14 50/200). Contrato OK.

**B1 lanzado:** job **`784465`** = `learnable`, job **`784466`** = `random`; ambos corriendo en `nodei-gpu-2`
(una GPU cada uno).
Runs W&B de B1: `learnable` = **`m7zvn949`**, `random` = **`h41dhdmn`**. URLs ya puestas en los `dam2_cache.yaml` de
cada carpeta (`model-m7zvn949/best` y `model-h41dhdmn/best`); solo valen cuando B1 termine `COMPLETED`.

### 15.1 Resultado B1 (DAM 2), etapa 3

| | `learnable` (job 784465, run `m7zvn949`) | `random` (job 784466, run `h41dhdmn`) |
|---|---|---|
| Estado | COMPLETED 0:0 | COMPLETED 0:0 |
| Tiempo | 6 min 54 s | 6 min 14 s |
| RAM (MaxRSS) | ~15.9 GB | ~16.4 GB |
| AUROC test total | **0.996** | **0.426** |
| AUROC depresión | 0.997 | 0.370 |
| AUROC ansiedad | 0.994 | 0.501 |
| `total_loss` test | 22.02 | 32.09 |

**Lectura.** DAM 2 aprende cuando hay relación entre audio/texto y etiqueta (AUROC ~1) y no aprende cuando no la hay
(AUROC ~0.5): no hay fuga entre splits y la cadena de entrenamiento/evaluación funciona. Con 40 audios de test el
AUROC varía bastante por azar: 0.37 en depresión con datos aleatorios es ruido, no un patrón.

**Límite.** La señal sintética es fácil a propósito (el texto trae palabras que dependen directamente del rango de
PHQ/GAD, y el audio un tono y una banda que dependen del puntaje). Un AUROC ~1 aquí prueba el pipeline, no dice nada
de cómo le irá con voz real. Tampoco sabemos todavía qué rama (audio o texto) aportó la señal.

**Costo:** con 200 audios × 8 épocas, ~6–7 min por corrida (vs 3.4 min con 16 audios × 3 épocas).

### 15.2 Resultado B2 (caché), etapa 3

| | `learnable` (job 784467, run `f3brbzbb`) | `random` (job 784468, run `idiokrs1`) |
|---|---|---|
| Estado | COMPLETED 0:0 | COMPLETED 0:0 |
| Tiempo / RAM | 1 min 47 s / ~11.6 GB | 2 min 25 s / ~11.7 GB |
| `Global cache size` | 280 | 280 |

URLs puestas en los `dam3_phase1.yaml` de cada carpeta: `backbone_cache-f3brbzbb/v0` (learnable) y
`backbone_cache-idiokrs1/v0` (random).

### 15.3 Resultado B3 (fase 1, `WhisperLLMA`), etapa 3

| | `learnable` (job 784469, run `a9s312dd`) | `random` (job 784470, run `wo4xu03t`) |
|---|---|---|
| Estado | COMPLETED 0:0 | COMPLETED 0:0 |
| Tiempo / RAM | 5 min 04 s / ~17.5 GB | 4 min 58 s / ~15.3 GB |
| Audios fuera del caché | 0 (train, val, test) | 0 (train, val, test) |
| MSE test (`total_loss`) | **0.0029** | **0.0031** |

**Lectura.** Las dos imitan muy bien la mitad BERT del caché (MSE ~0.003, frente a 0.136 en la etapa 1 con 16 audios y
3 épocas). Es lo esperado: la fase 1 no usa etiquetas, solo aprende a reproducir con el audio lo que BERT sacó del
texto, así que ambas pruebas se ven iguales aquí. Ojo: una MSE tan baja también puede deberse a que los vectores de
BERT de estos textos sintéticos (todos con el mismo vocabulario de relleno) se parecen mucho entre sí; si la MSE
refleja información útil sobre la etiqueta solo se ve en la fase 2.

URLs puestas en los `dam3_phase2.yaml`: `model-a9s312dd/best` (learnable) y `model-wo4xu03t/best` (random).

### 15.4 Resultado B4 (fase 2), etapa 3

| | `learnable` (job 784471, run `g83unt8l`) | `random` (job 784472, run `jokgpxfk`) |
|---|---|---|
| Estado | COMPLETED 0:0 | COMPLETED 0:0 |
| Tiempo / RAM | 6 min 14 s / ~18.6 GB | 4 min 15 s / ~17.4 GB |
| AUROC test total | **0.985** | 0.630 |
| AUROC depresión | 0.976 | 0.517 |
| AUROC ansiedad | 0.995 | **0.805** |
| `total_loss` test | 24.03 | 33.13 |

**`learnable`: DAM 3 aprende de punta a punta.** La fase 2 solo ve audio (dos Whisper congelados + cabeza) y llega a
AUROC 0.985.

**`random`: el control de ansiedad no salió limpio (0.805). Diagnóstico.** Se regeneró localmente el conjunto
`random` con la misma semilla (0) y se midió la correlación entre la etiqueta GAD y la señal que el generador pone en
audio y texto (el "latente"):

| Split | corr(gad, latente) | corr(phq, latente) |
|---|---|---|
| train (200) | -0.03 | -0.04 |
| val (40) | **-0.48** | -0.14 |
| test (40) | **-0.36** | +0.06 |

En modo `random` etiqueta y latente se sortean de forma independiente, pero con 40 audios y la semilla 0 val y test
quedaron, por azar, con una correlación negativa fuerte en GAD (en 2,000 simulaciones con n = 40 la desviación típica
es 0.16; -0.48 es muy poco probable, pero fue lo que salió). Cualquier modelo cuya salida dependa del latente, aunque
sea con una pendiente débil aprendida en train (-0.03), ordena bien el test. Por eso ansiedad da 0.805 y depresión
(sin esa correlación) da 0.517.

**Conclusión:** no es una fuga entre splits del pipeline (train no tiene relación y las etiquetas de test no se usan
en ningún paso), es un control mal sorteado. Hay que repetir el control con otra semilla verificada (correlación
cercana a 0 en todos los splits) antes de dar el resultado por bueno. Lección: el generador debería reportar esta
correlación.

## 16. DAM 1 con datos reales: baseline BRIDGE-MH (Bukavu), todas las versiones (2026-11-16)

Decisión (Sebas): usar **todas** las versiones del cuestionario (v1, v2, v4, v5, v6; la v3 está vacía) y un split
**aleatorio por persona** 70/15/15, estratificado por versión y PHQ > 9 (semilla 0). Las dos grabaciones de una persona
quedan siempre en el mismo split.

### 16.1 Datos (en el Mac, `KitsugiCode/Bridge_baseline/`)
- `data/raw/`: `Baseline 2.dta` + zips del portal (Binary = audio m4a; Tabular/STATA = respuestas). No se tocan.
- `data/clean/baseline_clean.{parquet,pkl}`: el `.dta` + `phq` (suma de 9 ítems, 0–27), `gad` (suma de 7, 0–21) e
  `in_analysis` (consentimiento de voz + al menos un audio + PHQ/GAD completos) → 4.440 personas.
- `hpc/bridge_prepare.py audio`: convierte los 6.305 audios de esas personas a WAV 16 kHz mono 16-bit (ffmpeg) en
  `data/clean/dam1/audio/<interview__key>/<AudioN__|AudioN_f__>.wav` (13 GB).
- `hpc/bridge_prepare.py csv`: escribe `train/val/test.csv` con `uuid` (por grabación), `filename`, `user`
  (= `interview__key`), `phq`, `gad`, `version` y la duración medida (`prompt_vad_trimmed_duration` en train,
  `prompt_vad_cut_duration` en val/test). Sin nombres ni teléfonos.

| split | personas | grabaciones | < 30 s | PHQ > 9 | GAD > 9 |
|---|---|---|---|---|---|
| train | 3.108 | 4.415 | 5 | 30.3 % | 25.1 % |
| val | 666 | 945 | 1 | 30.5 % | 25.7 % |
| test | 666 | 945 | 4 | 30.2 % | 24.9 % |

Audios < 30 s: 10 archivos de 8 personas; solo 3 personas quedan sin ningún audio ≥ 30 s (train PHQ 2; test PHQ 0 y 9).
En train no se pierden: `merge_rows_by_subject` suma la duración de las grabaciones de cada persona.

### 16.2 Config `hpc/configs/dam1_bridge.yaml`
Copia de `ordinal_regression_mtl_mar_2024/config.yaml` (modelo, LoRA, pérdidas y optimizador sin cambios). Cambios:
rutas a los CSV/audio en Hypatia, `merge_rows_by_subject: many_to_one` en train (como upstream), `effective_batch_size`
32 (upstream 128; ~97 pasos por época), `training_examples_per_eval` 3108 (una validación por época), `max_epochs` 32 y
`early_stopping_patience` 10 (upstream).

Notas: el backbone es `whisper-small.en` (inglés) y el habla es en lenguas locales/swahili/francés (idioma no
registrado); el perfil log-mel es el de Kintsugi. Es una primera corrida de referencia.

### 16.3 Primer intento (job 803172): falla en train por `merge_rows_by_subject`
Arrancó bien (W&B run `3p9rb800`; filtró por < 30 s: 1 de 3.108 en train, 1 de 945 en val, 4 de 945 en test) y falló
al empezar la primera época: `AttributeError: 'Series' object has no attribute 'audio_filename'`.
Causa: `kirad.dataset_utils.merge_df_rows_by_subject` (lo usa `merge_rows_by_subject: many_to_one`) solo conserva las
columnas de sus listas `common_cols` / `list_cols` / `sum_cols`, y no incluye `audio_filename` ni `text_filename`, que
`load_relative_dataset` crea a partir de `filename` + `audio_root`. Al fusionar las filas de cada persona se pierden
las rutas de audio. Arreglo (2 líneas): agregar `audio_filename` y `text_filename` a `list_cols`. Probado con
`train.csv`: 4.415 filas → 3.108 personas, cada una con la lista de sus WAV y la duración sumada.

### 16.4 Segunda corrida: Whisper multilingüe
`hpc/configs/dam1_bridge_multilingual.yaml` = `dam1_bridge.yaml` cambiando solo el backbone de `openai/whisper-small.en`
(inglés) a `openai/whisper-small` (multilingüe; mismo tamaño, mismas 80 bandas log-mel) y el `experiment_name`.
Mismos datos, split, hiperparámetros y semilla, para comparar directamente con la corrida en inglés (job 803180).

### 16.5 Resultados (test: 941 grabaciones de 666 personas)
Ambos jobs `COMPLETED` (exit 0:0), sin errores; los dos se detuvieron por early stopping antes de 32 épocas.

| métrica (test) | 803180 `whisper-small.en` (W&B `pnybari4`) | 803200 `whisper-small` multilingüe (W&B `bnkqcsas`) |
|---|---|---|
| tiempo | 2:51:12 | 2:07:00 |
| AUROC@0%i global | **0.666** | 0.655 |
| AUROC depresión | 0.656 | 0.646 |
| AUROC ansiedad | 0.675 | 0.664 |
| AUROC@40%i global | 0.734 | 0.716 |
| sn_eq_sp@0%i global | 0.607 | 0.604 |
| sn_eq_sp@0%i depresión / ansiedad | 0.607 / 0.608 | 0.600 / 0.608 |
| total_loss | 21.54 | 23.34 |

Lectura: hay señal por encima del azar (AUROC ~0.66) con split aleatorio por persona. La diferencia entre inglés y
multilingüe (~0.01 de AUROC) está dentro del ruido esperable con ~940 grabaciones de test. Pendiente: evaluación con
encuestadores fuera (efecto encuestador) para saber cuánto de esta señal es voz y cuánto es encuestador/dispositivo.

### 16.6 Historial de entrenamiento (W&B)
Descargado en Hypatia con `python hpc/wandb_history.py pnybari4 bnkqcsas --out ~/Kintusgi/results`
(`wandb.Api().run(...)`; guarda `<run>_val.csv` y `<run>_train.csv`). ~98 pasos de optimizador por época, una validación por época.

| | small.en (`pnybari4`) | small multilingüe (`bnkqcsas`) |
|---|---|---|
| validaciones (épocas) | 23 | 17 |
| mejor checkpoint (máx. val sn_eq_sp@0%i) | validación 13 (paso 1274), 0.660 | validación 7 (paso 686), 0.654 |
| val AUROC en el mejor: global / dep / anx | 0.709 / 0.706 / 0.711 | 0.701 / 0.703 / 0.698 |
| test AUROC: global / dep / anx | 0.666 / 0.656 / 0.675 | 0.655 / 0.646 / 0.664 |

La AUROC de validación sube a ~0.68 en 3 épocas y luego se estanca (0.66–0.71). La pérdida de train baja de forma
continua (28.5 → 17.2) mientras la de validación se aplana desde ~la época 13 (inicio de sobreajuste). La caída
validación → test (~0.04) es la esperable al elegir el máximo de una métrica ruidosa.

## 17. DAM 1 solo con la versión 6 del cuestionario (2026-11-23)

Decisión (Sebas, tras la reunión con Ignacio): repetir todo solo con la **versión 6** (cuestionario final, dos
grabaciones por persona).

- `hpc/bridge_subset.py`: filtra los CSV de `dam1/` a `version == v6` en `~/Kintusgi/Data/bridge_baseline/dam1_v6/`
  (el audio no se copia; `audio_root` sigue apuntando a `dam1/audio`). Se hereda el split por persona, así que la
  parte v6 de cada split es un 70/15/15 de las personas v6 estratificado por PHQ > 9.

| split | personas | grabaciones | PHQ > 9 | GAD > 9 | GAD > 14 (personas) |
|---|---|---|---|---|---|
| train | 1.275 | 2.552 | 23.8 % | 21.4 % | 58 |
| val | 273 | 547 | 23.8 % | 18.3 % | 14 |
| test | 273 | 546 | 23.8 % | 19.4 % | 10 |

- Configs: `dam1_bridge_v6.yaml` (`whisper-small.en`, referencia) y `dam1_bridge_v6_large.yaml`
  (`openai/whisper-large-v2`). Se usa large-v2 porque mantiene 80 bandas log-mel; large-v3 usa 128 y exigiría cambiar
  el extractor de características y el perfil log-mel. `training_examples_per_eval` = 1275 (una validación por época);
  el resto igual a `dam1_bridge.yaml`.
- Ojo: con 273 personas en test, GAD > 14 tiene solo 10 personas positivas; esa métrica será muy ruidosa.

### 17.1 Comparativa con todas las versiones
Mismo protocolo con **todas las versiones** (v1, v2, v4, v5, v6), para comparar con v6:
- small.en y small multilingüe: ya corridos en la sección 16 (jobs 803180 → W&B `pnybari4`, 803200 → `bnkqcsas`);
  no se repiten, solo se evalúan con las métricas nuevas (17.3).
- large-v2: nueva config `dam1_bridge_large.yaml` (= `dam1_bridge.yaml` cambiando solo el modelo y el
  `experiment_name`).

Experimentos de esta sección:

| # | datos | Whisper | config |
|---|---|---|---|
| A | todas las versiones | small.en | `dam1_bridge.yaml` (803180, `pnybari4`) |
| B | todas las versiones | small (multilingüe) | `dam1_bridge_multilingual.yaml` (803200, `bnkqcsas`) |
| C | todas las versiones | large-v2 | `dam1_bridge_large.yaml` |
| D | solo v6 | small.en | `dam1_bridge_v6.yaml` |
| E | solo v6 | large-v2 | `dam1_bridge_v6_large.yaml` |

Límite de tiempo: los jobs large se lanzan con `sbatch --time` mayor que el de `run_job.sbatch` (12 h), porque
large-v2 (~635 M parámetros en el encoder) es ~6 veces más pesado que small (~88 M). Con 2 GPUs por usuario, el
tercer job queda en cola hasta que se libere una.

### 17.2 CORAL y de dónde salen los scores
- **CORAL** (`kirad/losses.py`, `CORALLoss`; Cao et al. 2019, arXiv:1901.07884) es regresión ordinal: cada tarea
  (depresión, ansiedad) produce **un solo logit x** por ventana de 30 s, y la pérdida aprende un sesgo b_k por cada
  posible corte k (27 para PHQ, 21 para GAD). La probabilidad de estar por encima del corte k es
  `P(y > k) = sigmoid(x + b_k)`, y la pérdida es la suma de entropías binarias de esos K-1 problemas
  ("código termómetro"). A diferencia de softmax, respeta el orden: el mismo x sirve para todos los cortes.
- **Score.** Lo que el pipeline guarda y evalúa es x promediado sobre las ventanas de cada grabación
  (`average_and_variance_tensor_in_segments` en `OrdinalRegressionPLModule.step`). Al final de cada validación y del
  test lo sube a W&B como artifact `scores-<tarea>-<split>-<run_id>` (tabla `scores`: uuid, scores,
  score_variances, labels, quantized_labels; `on_validation_test_epoch_end`, `kirad/base_models.py`). Hay una
  versión por validación (v0, v1, ...) y una para test.
- **Probabilidades.** `hpc/bridge_eval.py` lee los sesgos b_k del mejor checkpoint (`chkpts/<experiment_name>/
  epoch-*.ckpt`, claves `tasks.<tarea>.loss.biases`) y calcula `prob_gt<c> = sigmoid(score + b_c)` para cada corte.
  La AUROC y los umbrales no cambian (es una transformación monótona); la probabilidad es más fácil de interpretar.
- **Umbral y matriz de confusión** (`hpc/bridge_eval.py`): por tarea y corte, umbral donde sensibilidad =
  especificidad en **validación** (tabla de la mejor época = versión `v<índice de la mejor validación>`), aplicado a
  **test**: TP, FP, FN, TN, sensibilidad (= recall), especificidad, precisión (PPV), NPV, F1 y accuracy. Se calcula
  por grabación (como el pipeline) y por persona (promedio de los scores de sus grabaciones).
- **Errores por grupo** (`hpc/bridge_error_analysis.py`, en el Mac porque usa la base de la encuesta): por persona en
  test y por grupo (sexo, edad, lengua materna, desplazado, educación, orden de las preguntas): FP, FN, tasas de
  falsos positivos/negativos y la correlación de Spearman entre el score y el PHQ/GAD real. También escribe la base
  limpia + scores (`<run>_baseline_with_scores.pkl/.csv`) para Ignacio: solo personas de val/test tienen score.

### 17.3 Lanzamiento y primeros fallos (2026-11-23)
**Partición `gpu_economia`.** Sebas tiene acceso (grupo `gpu_economia`). Nodo `nodea-gpu-3`: 2 × NVIDIA **L40, 46 GB**
(driver 575), 64 CPUs, ~1 TB RAM. El nodo está bajo la reserva `root_30` (grupo `gpu_economia`, hasta
2026-11-25 14:48), así que los jobs se envían con `-p gpu_economia --reservation=root_30`. El límite por usuario
(QoS `gpu`) es el mismo en ambas particiones: **2 GPUs, 32 CPUs, ~187 GB** en total; el tercer job queda en cola
(`QOSMaxGRESPerUser`). Los large-v2 van a `gpu_economia` (más memoria y más rápida); small.en a `gpu`.

Primer envío (jobs 810585 E, 810586 D, 810587 C) → los tres FAILED:
- **E y C (gpu_economia):** `ImportError: Failed to initialize: Bad git executable`. El nodo `nodea-gpu-3` no tiene
  `git` (ni módulo). `kirad/utils.py` importa GitPython y `get_package_info` lee el commit con él.
  Arreglo: en `hpc/run_job.sbatch`, si no hay `git`, se exporta `GIT_PYTHON_REFRESH=quiet` y el commit se lee de
  `.git/HEAD` + la ref (sin ejecutable) en `KIRAD_GIT_COMMIT`; en `kirad/utils.py` (`get_package_info`) se añadió un
  `except Exception` que usa `KIRAD_GIT_COMMIT` en vez de fallar. No se instaló nada en el entorno conda.
- **D (gpu):** `ValueError: Invalid training_examples_per_eval: 1275 is not divisible by global_batch_size = 2`.
  Arreglo: 1276 en `dam1_bridge_v6.yaml` y `dam1_bridge_v6_large.yaml`.

### 17.4 Métricas nuevas de A y B (todas las versiones, small.en y multilingüe)
`python hpc/bridge_eval.py pnybari4 bnkqcsas` (resultados en `~/Kintusgi/results/<run>_{scores,confusion}.csv`).
Primer intento falló: tomaba el índice de la fila del historial de W&B (`v4072`) en vez del índice de la validación;
corregido (commit 917a8e8) contando solo las filas con métrica de validación y verificando que coincida con el
número de artifacts de scores de validación. La tabla de validación usada es la del mejor checkpoint: `v12` (A,
validación 13 de 23) y `v6` (B, validación 7 de 17). Sesgos CORAL leídos de `chkpts/<experiment_name>/epoch-*.ckpt`.

Control: el promedio por corte de sensibilidad/especificidad coincide con `sn_at_val_eq_thresh` /
`sp_at_val_eq_thresh` que reporta kirad (A depresión: (0.635 + 0.579)/2 = 0.607).

Test por grabación (941), umbral de validación sn = sp:

| exp | tarea | corte | positivos | TP | FP | FN | TN | sens. (recall) | espec. | precisión | NPV | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A small.en | dep | >9 | 266 | 169 | 271 | 97 | 404 | 0.635 | 0.599 | 0.384 | 0.806 | 0.479 | 0.609 |
| A | dep | >14 | 107 | 62 | 320 | 45 | 514 | 0.579 | 0.616 | 0.162 | 0.919 | 0.254 | 0.612 |
| A | anx | >4 | 496 | 326 | 167 | 170 | 278 | 0.657 | 0.625 | 0.661 | 0.621 | 0.659 | 0.642 |
| A | anx | >9 | 219 | 134 | 297 | 85 | 425 | 0.612 | 0.589 | 0.311 | 0.833 | 0.412 | 0.594 |
| A | anx | >14 | 46 | 24 | 272 | 22 | 623 | 0.522 | 0.696 | 0.081 | 0.966 | 0.140 | 0.688 |
| B multilingüe | dep | >9 | 266 | 155 | 250 | 111 | 425 | 0.583 | 0.630 | 0.383 | 0.793 | 0.462 | 0.616 |
| B | dep | >14 | 107 | 62 | 287 | 45 | 547 | 0.579 | 0.656 | 0.178 | 0.924 | 0.272 | 0.647 |
| B | anx | >4 | 496 | 294 | 169 | 202 | 276 | 0.593 | 0.620 | 0.635 | 0.577 | 0.613 | 0.606 |
| B | anx | >9 | 219 | 118 | 275 | 101 | 447 | 0.539 | 0.619 | 0.300 | 0.816 | 0.386 | 0.600 |
| B | anx | >14 | 46 | 29 | 283 | 17 | 612 | 0.630 | 0.684 | 0.093 | 0.973 | 0.162 | 0.681 |

Test por persona (A, 664 personas; score = promedio de sus grabaciones): dep >9 sens 0.632 / espec 0.605 /
precisión 0.410 / F1 0.497; dep >14 0.605 / 0.602 / 0.184 / 0.283; anx >4 0.639 / 0.635 / 0.685 / 0.661;
anx >9 0.618 / 0.577 / 0.326 / 0.427; anx >14 0.583 / 0.699 / 0.100 / 0.171.

Lectura: con ~60 % de sensibilidad y especificidad, la **precisión** cae cuando el corte es raro (dep >14: 16 %;
anx >14: 8 %): la mayoría de los positivos predichos son falsos positivos. El **NPV** es alto (0.81–0.97) porque la
mayoría de la gente está por debajo del corte. Es la consecuencia esperable del desbalance que señaló Ignacio.

### 17.5 Segundo envío (2026-11-23): memoria del host en large-v2
Jobs 810595 (E, v6 large-v2, W&B `sefd1s0f`) y 810596 (C, todas large-v2, `hndj4w4v`) arrancaron bien en las L40
(GPU ~95 %, ~18 GB de 46 GB) y 810597 (D, v6 small.en) quedó en cola.
- **E falló a los 8 min** al guardar el primer checkpoint: `RuntimeError: ... inline_container.cc ... unexpected pos`
  seguido de `DefaultCPUAllocator: can't allocate memory`. El job pedía 32 GB de RAM (default de `run_job.sbatch`);
  el proceso ya usaba ~22 GB y la serialización del checkpoint de large-v2 superó el límite del cgroup.
- C usaba ~22 GB y habría fallado igual en su primer checkpoint: se canceló (`scancel 810596`, 9 min).
- Relanzados con `--mem=72G` (límite QoS por usuario: ~187 GB en total; 72 + 72 + 32 de D cabe):
  **810610 (E)** corriendo en `nodea-gpu-3`; **810611 (C)** en cola hasta que D termine; **810597 (D)** corriendo
  en `nodei-gpu-1` (gpu, 32 GB basta para small).

### 17.6 Análisis de errores por grupo demográfico (A y B, todas las versiones)
Scores y matrices copiados de Hypatia al Mac (`scp`) a `Bridge_baseline/data/clean/results/` y analizados en el Mac
con `python hpc/bridge_error_analysis.py --base Bridge_baseline --results Bridge_baseline/data/clean/results
--run <run>` (la base de la encuesta no sale del Mac). Cambio: la correlación de Spearman se calcula con rangos +
Pearson porque el entorno del Mac no tiene scipy. Nivel persona, test (664 personas), umbral de validación sn = sp.

Salidas (no van al repo, tienen datos de la encuesta): `<run>_errors_by_group.csv` y
`<run>_baseline_with_scores.{pkl,csv}` = base limpia (4.488 filas) + `depression_score`, `anxiety_score`
(promedio por persona del score CORAL), `*_split` (val/test) y `*_pred_gt<c>` (0/1 con el umbral de validación).
Solo 1.330 personas (val + test) tienen score; las de train no (el modelo las vio).

Depresión, PHQ > 9, A (small.en) — entre paréntesis B (multilingüe):

| grupo | personas | positivos | Spearman score~PHQ | sensibilidad | especificidad | precisión |
|---|---|---|---|---|---|---|
| mujeres | 580 | 174 | 0.36 (0.30) | 0.66 (0.63) | 0.61 (0.62) | 0.42 (0.41) |
| hombres | 84 | 27 | 0.09 (0.05) | 0.48 (0.26) | 0.58 (0.74) | 0.35 (0.32) |
| 18–24 años | 197 | 45 | 0.26 (0.16) | 0.38 (0.36) | 0.74 (0.76) | 0.30 (0.30) |
| 25–34 | 235 | 68 | 0.26 (0.19) | 0.59 (0.47) | 0.61 (0.68) | 0.38 (0.37) |
| 35–44 | 105 | 39 | 0.50 (0.41) | 0.87 (0.87) | 0.52 (0.47) | 0.52 (0.49) |
| 45+ | 119 | 44 | 0.16 (0.16) | 0.75 (0.73) | 0.39 (0.41) | 0.42 (0.42) |
| educación superior | 131 | 37 | 0.23 (0.20) | 0.46 (0.41) | 0.74 (0.83) | 0.41 (0.48) |
| secundaria | 381 | 107 | 0.28 (0.20) | 0.56 (0.50) | 0.61 (0.64) | 0.36 (0.36) |
| primaria | 103 | 35 | 0.33 (0.31) | 0.86 (0.83) | 0.46 (0.40) | 0.45 (0.41) |
| sin educación | 43 | 21 | 0.45 (0.38) | 0.90 (0.81) | 0.36 (0.41) | 0.58 (0.57) |
| desplazados | 152 | 56 | 0.36 (0.34) | 0.70 (0.59) | 0.57 (0.60) | 0.49 (0.46) |
| no desplazados | 512 | 145 | 0.31 (0.23) | 0.61 (0.57) | 0.61 (0.64) | 0.38 (0.39) |
| shi | 473 | 143 | 0.30 (0.26) | 0.66 (0.61) | 0.59 (0.59) | 0.41 (0.39) |
| havu | 41 | 11 | 0.54 (0.38) | 0.64 (0.55) | 0.60 (0.80) | 0.37 (0.50) |
| tembo | 23 | 8 | 0.80 (0.73) | 0.88 (0.62) | 0.80 (0.73) | 0.70 (0.56) |
| "other" | 113 | 35 | 0.28 (0.16) | 0.46 (0.46) | 0.64 (0.72) | 0.36 (0.42) |

Ansiedad GAD > 9 (A) sigue el mismo patrón: hombres sens. 0.48 vs mujeres 0.65; 35–44 y 45+ sens. 0.76/0.69 con
espec. 0.42/0.31; sin educación sens. 0.92 / espec. 0.32; educación superior 0.60 / 0.75.

Correlaciones en test (Spearman, persona): score de depresión vs PHQ real 0.33 (A) / 0.26 (B); **score vs edad
0.34 / 0.32**, mientras que PHQ real vs edad es solo 0.17; **score vs educación −0.33 / −0.38**, PHQ real vs
educación −0.13. Score de ansiedad vs GAD real 0.34 / 0.27.

Lectura:
- **El modelo empuja el score hacia arriba en personas mayores y con menos educación**, más de lo que esos grupos
  tienen realmente de depresión: ahí detecta casi todo (sens. 0.75–0.90) pero con muchos falsos positivos
  (espec. 0.36–0.52). En jóvenes y con educación superior hace lo contrario (sens. 0.38–0.46). Es decir, parte de
  la señal parece ser "edad / educación en la voz", no síntomas.
- **En hombres el modelo casi no ordena** (Spearman 0.09 / 0.05; 84 hombres en test).
- Las lenguas pequeñas (tembo, havu, fuliru, bembe, nyindu) tienen muy pocas personas: no se puede concluir nada.
- Pendiente: repetir este análisis con C, D y E, y separar el efecto de edad/educación (p. ej. AUROC dentro de cada
  grupo de edad, o un modelo solo con edad y educación como línea base).

### 17.7 Report card nativo de Kintsugi (`get_wandb_report_card_tsvs`)
`kirad.utils.get_wandb_report_card_tsvs` falla en nuestros runs con `KeyError: 'tasks'`: lee las tareas de
`run.config["model_params"]["tasks"]` y las rutas de los splits de `data_params["<split>_split_or_dataset_version"]`
/ `data_params["data_sources"]` (formato de config antiguo). Nuestras configs tienen las tareas en
`training_params.tasks` y los CSV en `data_params["metadata=..."][<split>]["df_or_dataset_version"]`.
- `hpc/bridge_report_card.py` (nuevo): **no modifica kirad**. Sustituye `wandb.Api` dentro de `kirad.utils` por una
  subclase cuyo `run()` agrega esas llaves al objeto de config en memoria (no se escribe en W&B) y llama la función
  original sin cambios. Uso: `KINTSUGI_RAD_PATH=$PWD PYTHONPATH=$PWD python hpc/bridge_report_card.py <ids>`.
- Efecto propio de la función: sube un artifact `report-cards-<run_id>` a cada run de W&B (wandb.init resume).
- Salida: `~/Kintusgi/results/<run>_report_card_<tarea>.tsv` (copiadas al Mac en `results/`).
- Control: sn_eq_sp_test de A depresión = 60.7 %, igual al 0.607 de `bridge_eval`.

Métricas multiclase (bandas de severidad, umbrales afinados en validación para macro-F1), test:

| exp | dep macro-F1 (base) | dep accuracy (base) | dep error abs. (base) | anx macro-F1 (base) | anx accuracy (base) | anx error abs. (base) |
|---|---|---|---|---|---|---|
| A | 0.426 (0.278) | 0.634 (0.717) | 0.486 (0.396) | 0.355 (0.161) | 0.440 (0.473) | 0.773 (0.755) |
| B | 0.398 (0.278) | 0.549 (0.717) | 0.550 (0.396) | 0.328 (0.161) | 0.420 (0.473) | 0.794 (0.755) |
| C | 0.458 (0.278) | 0.638 (0.717) | 0.484 (0.396) | 0.330 (0.161) | 0.410 (0.473) | 0.819 (0.755) |
| D | 0.418 (0.288) | 0.575 (0.761) | 0.523 (0.317) | 0.341 (0.177) | 0.499 (0.547) | 0.632 (0.685) |
| E | 0.424 (0.288) | 0.685 (0.761) | 0.448 (0.317) | 0.273 (0.177) | 0.466 (0.547) | 0.768 (0.685) |

"base" = mejor clasificador constante. Lectura: el macro-F1 supera la base; accuracy y error absoluto no (la
mayoría de la gente está en la banda más baja, y el modelo reparte personas entre bandas). Las filas de abstención
(indet) del report card no se usan: mezclan cortes y no coinciden con `@40%i` del log de test.

### 17.8 Resultados de C, D y E (2026-11-23)
Los tres terminaron COMPLETED: **810611 C** (todas, large-v2, W&B `qviai5hj`, 4 h 13, `nodea-gpu-3`),
**810597 D** (v6, small.en, `oxmqyl0h`, 1 h 44, `nodei-gpu-1`), **810610 E** (v6, large-v2, `jmy587yu`, 1 h 21,
`nodea-gpu-3`). En Hypatia: `bridge_eval.py`, `wandb_history.py` y `bridge_report_card.py` sobre los tres; todo
copiado al Mac (`scp`) a `Bridge_baseline/data/clean/results/`.

| exp | validaciones (mejor) | checkpoint | test AUROC dep / anx (grabación, W&B) | sn=sp dep / anx |
|---|---|---|---|---|
| A | 23 (13) | epoch-13 | 0.656 / 0.675 | 0.607 / 0.608 |
| B | 17 (7) | epoch-7 | 0.646 / 0.664 | 0.600 / 0.608 |
| C | 19 (9) | epoch-9 | 0.666 / 0.679 | 0.610 / 0.619 |
| D | 31 (27) | epoch-27 | 0.702 / 0.692 | 0.627 / 0.618 |
| E | 14 (4) | epoch-4 | 0.607 / 0.591 | 0.535 / 0.550 |

Nota de datos: en v6, kirad descarta por duración a 1 persona de test (`70-64-56-30`, 2 grabaciones) y 1 grabación
más: se puntúan 272 personas / 543 grabaciones (igual en todas las corridas).

Historial (W&B): E tiene su mejor AUROC de validación en la época 1 (0.666) y luego oscila entre 0.58 y 0.66; el
early stopping (sn = sp, paciencia 10) lo cortó en la época 14 **con las pérdidas aún bajando** (train 27.6 → 23.6,
val 26.1 → 23.0): es sub-entrenamiento con una validación ruidosa (273 personas), no sobreajuste. D mejora lento y su
mejor checkpoint es la validación 27 de 31; su pérdida de validación se aplana en ~21.1 desde la época 25.
large-v2 tarda ~1.7× más por época que small aun en la L40 (C 13.3 min/época vs A ~7.8; E 5.8 vs D 3.4).

Matriz de confusión por persona, test, umbral de validación sn = sp (A y B en 17.4):

| exp | tarea | corte | pers. | pos. | TP | FP | FN | TN | recall | espec. | precisión | NPV | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | dep | >9 | 664 | 201 | 116 | 175 | 85 | 288 | 0.577 | 0.622 | 0.399 | 0.772 | 0.472 |
| C | dep | >14 | 664 | 86 | 51 | 192 | 35 | 386 | 0.593 | 0.668 | 0.210 | 0.917 | 0.310 |
| C | anx | >4 | 664 | 368 | 230 | 109 | 138 | 187 | 0.625 | 0.632 | 0.678 | 0.575 | 0.651 |
| C | anx | >9 | 664 | 165 | 89 | 190 | 76 | 309 | 0.539 | 0.619 | 0.319 | 0.803 | 0.401 |
| C | anx | >14 | 664 | 36 | 24 | 204 | 12 | 424 | 0.667 | 0.675 | 0.105 | 0.972 | 0.182 |
| D | dep | >9 | 272 | 65 | 38 | 75 | 27 | 132 | 0.585 | 0.638 | 0.336 | 0.830 | 0.427 |
| D | dep | >14 | 272 | 21 | 13 | 93 | 8 | 158 | 0.619 | 0.629 | 0.123 | 0.952 | 0.205 |
| D | anx | >4 | 272 | 123 | 82 | 57 | 41 | 92 | 0.667 | 0.617 | 0.590 | 0.692 | 0.626 |
| D | anx | >9 | 272 | 53 | 32 | 80 | 21 | 139 | 0.604 | 0.635 | 0.286 | 0.869 | 0.388 |
| D | anx | >14 | 272 | 10 | 6 | 70 | 4 | 192 | 0.600 | 0.733 | 0.079 | 0.980 | 0.140 |
| E | dep | >9 | 272 | 65 | 31 | 67 | 34 | 140 | 0.477 | 0.676 | 0.316 | 0.805 | 0.380 |
| E | dep | >14 | 272 | 21 | 8 | 79 | 13 | 172 | 0.381 | 0.685 | 0.092 | 0.930 | 0.148 |
| E | anx | >4 | 272 | 123 | 62 | 49 | 61 | 100 | 0.504 | 0.671 | 0.559 | 0.621 | 0.530 |
| E | anx | >9 | 272 | 53 | 24 | 79 | 29 | 140 | 0.453 | 0.639 | 0.233 | 0.828 | 0.308 |
| E | anx | >14 | 272 | 10 | 4 | 74 | 6 | 188 | 0.400 | 0.718 | 0.051 | 0.969 | 0.091 |

Por grabación están en `<run>_confusion.csv` (`level == recording`).

### 17.9 Comparativa justa: mismas personas (`hpc/bridge_compare.py`, nuevo, corre en el Mac)
A, B y C se prueban en 664 personas y D, E en 273: no son comparables directamente. Como el split v6 es un
subconjunto del split de todas las versiones, las 272 personas v6 de test (las puntuadas por todos) son personas de
test de los cinco runs. El script calcula, por persona: AUROC por corte con IC 95 % bootstrap sobre personas (1.000
remuestreos, semilla 0), en el test propio de cada run y en las 272 personas v6; la diferencia pareada de cada run
contra A (mismos remuestreos); y correlaciones de Spearman (sin scipy: rangos + Pearson). Salidas:
`compare_auroc.csv`, `compare_paired.csv`, `compare_corr.csv` en `results/`.

AUROC por persona en las 272 personas v6 (positivos entre paréntesis):

| corte | A | B | C | D | E |
|---|---|---|---|---|---|
| PHQ > 9 (65) | 0.671 | 0.647 | 0.650 | 0.681 | 0.605 |
| PHQ > 14 (21) | 0.641 | 0.650 | 0.643 | 0.720 | 0.601 |
| GAD > 4 (123) | 0.685 | 0.656 | 0.661 | 0.698 | 0.624 |
| GAD > 9 (53) | 0.633 | 0.617 | 0.609 | 0.672 | 0.556 |
| GAD > 14 (10) | 0.564 | 0.617 | 0.604 | 0.702 | 0.569 |

Diferencia pareada contra A (IC 95 %): D: PHQ>9 +0.009 (−0.04, +0.06); PHQ>14 +0.080 (−0.00, +0.16); GAD>4 +0.013
(−0.03, +0.06); GAD>9 +0.039 (−0.02, +0.09); GAD>14 +0.138 (+0.01, +0.26). E: PHQ>9 −0.067 (−0.14, +0.00); GAD>4
−0.061 (−0.12, −0.00). B y C: −0.02 a −0.03 en los cortes principales, IC incluyen 0.

En el test propio (personas, IC 95 %): A PHQ>9 0.660 (0.62–0.70); C 0.662 (0.61–0.71); D 0.681 (0.60–0.76);
E 0.605 (0.52–0.69).

Lectura:
- En los cortes principales (PHQ > 9, GAD > 4) **D y A empatan** sobre las mismas personas: entrenar solo con v6 no
  gana casi nada en personas v6. La ventaja de D está en los cortes raros, con 10–21 positivos: sugerente, no firme.
- **large-v2 no ayuda**: C ≈ A con todas las versiones; E queda por debajo de A en todo.

### 17.10 Errores por grupo demográfico y correlaciones: C, D y E
`bridge_error_analysis.py --run <id>` para C, D y E (salidas `<run>_errors_by_group.csv` y
`<run>_baseline_with_scores.{pkl,csv}`; en D y E tienen score 545 personas, val + test v6).

Spearman, test propio, por persona (depresión; ansiedad entre paréntesis):

| exp | score~PHQ real | score~edad | score~educación | PHQ real~edad | PHQ real~educación |
|---|---|---|---|---|---|
| A | 0.33 (0.34) | 0.34 (0.35) | −0.33 (−0.32) | 0.17 (0.14) | −0.13 (−0.09) |
| B | 0.26 (0.28) | 0.32 (0.34) | −0.38 (−0.37) | 0.17 (0.14) | −0.13 (−0.09) |
| C | 0.31 (0.31) | 0.36 (0.36) | −0.38 (−0.37) | 0.17 (0.14) | −0.13 (−0.09) |
| D | 0.28 (0.29) | 0.27 (0.29) | −0.20 (−0.19) | 0.19 (0.11) | −0.14 (−0.12) |
| E | 0.23 (0.19) | 0.36 (0.37) | −0.35 (−0.35) | 0.19 (0.11) | −0.14 (−0.12) |

PHQ > 9 por grupo (recall / especificidad), D: 18–24 0.31 / 0.69; 25–34 0.57 / 0.71; 35–44 0.81 / 0.48;
45+ 0.57 / 0.48; sin educación 0.88 / 0.45; primaria 0.80 / 0.57; secundaria 0.47 / 0.63; superior 0.54 / 0.73.
Hombres (PHQ > 9), Spearman score~PHQ: A 0.09, B 0.05, C 0.13, D 0.37 (46 hombres), E 0.09; recall en hombres:
0.48, 0.26, 0.26, 0.60, 0.10.

Lectura: el patrón de 17.6 se repite en las cinco corridas (mayores y menos educados sobre-marcados, jóvenes y con
educación superior subdetectados). D es la que menos depende de edad/educación; E la que más (más edad que PHQ).

### 17.11 Presentación
`KitsugiCode/Avance_23_NOV_2026.pptx` (24 slides, mismo formato que la del 16 NOV): pendientes, cinco experimentos,
v6, cambios de código, nodo de economía, CORAL y scores, export para Ignacio, glosario, historial, AUROC, misma gente,
matrices de confusión, precisión por corte, report card, correlaciones, errores por edad/educación, hombres,
large vs small, caveats, conclusiones, siguientes pasos y reproducibilidad.

### 17.12 Gráficas de análisis y presentación del 25 NOV
`hpc/bridge_figures.py` (nuevo, corre en el Mac): todas las gráficas comparativas de A–E. **Regla de privacidad:**
solo agregados (medias por grupo, AUROC, conteos); ningún punto individual, boxplots sin outliers, grupos con
menos de 10 personas ocultos; nada con `interview__key` se imprime. Salidas en `KitsugiCode/Avance_25_NOV_figs/`
(17 PNG + `summary.json` con los números agregados). Presentación: `KitsugiCode/Avance_25_NOV_2026.pptx` (25 slides,
construida en el Mac con `Avance_25_NOV_figs/deck/build_25nov.py`).

Análisis nuevos (persona, test; IC 95 % bootstrap sobre personas, 500 remuestreos):
- **Versiones.** PHQ > 9: v1 44 %, v2 43 %, v4 39 %, v5 32 %, v6 24 %; desplazados 29/32/29/26/22 %. Edad mediana
  (28–29), % mujeres (86–93 %) y educación mediana (12–13 años) casi iguales. v6: 2 grabaciones por persona (secciones
  A y F, 62.2 / 62.4 s), v1–v5: 1.
- **Dos grabaciones v6 (D, 267 personas):** Spearman entre el score de A y el de F = 0.79; AUROC PHQ > 9 con solo A
  0.654, solo F 0.677, promedio 0.675.
- **Calibración** de P(> 9): media bien calibrada (A dep: 0.29 predicho vs 0.30 observado) pero comprimida: quintil
  superior 0.42 vs 0.47, inferior 0.22 vs 0.17. E casi plano (0.25 → 0.29).
- **Score por banda de PHQ:** la mediana sube de 0–4 a 15–19, pero 20–27 no queda por encima de 15–19.
- **Correlación parcial** score~PHQ quitando edad, educación, sexo y desplazamiento: A 0.33 → 0.27; D 0.28 → 0.23.
- **AUROC dentro de grupos (PHQ > 9, A):** 18–24 0.62, 25–34 0.64, 35–44 0.78, 45+ 0.54; hombres 0.51 (C 0.51,
  D 0.71 con 46 hombres); mujeres 0.68.
- **Línea base demográfica** (regresión logística sobre edad, años de educación, sexo, desplazamiento y versión,
  ajustada en train): PHQ > 9 0.626 (test de todas las versiones) vs voz A 0.660 y voz A + demografía 0.675 (combinación
  ajustada en validación); GAD > 9 0.574 vs 0.652 / 0.663. En v6: demografía 0.581, voz D 0.681, D + demografía 0.663
  (la combinación empeora: se ajusta con solo 273 personas de validación).
- **Acuerdo entre corridas** (Spearman de scores, 272 personas v6): A–B 0.90, A–C 0.87, A–D 0.78, D–E 0.49.
- **Entrevistador** (`Q1_enumeratorid`): % de la varianza del score entre entrevistadores: A 38 %, B 47 %, C 40 %,
  D 29 %, E 36 % (PHQ real: 25–27 %). AUROC PHQ > 9 ordenando solo dentro de cada entrevistador (score menos la media
  del entrevistador; entrevistadores con ≥ 5 personas de test): A 0.660 → 0.589, B 0.636 → 0.577, C 0.662 → 0.601,
  D 0.656 → 0.602, E 0.637 → 0.556. Parte de la señal es quién hizo la entrevista.

Lectura para los siguientes experimentos: split que deje fuera entrevistadores completos; reportar AUROC dentro de
grupos de edad/educación y contra la línea base demográfica; v6 sola no ordena mejor a la gente de v6 que todas las
versiones; large-v2 no aporta.
