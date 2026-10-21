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
