# Bitácora — kintsugi-hpc-uniandes

Registro de cada paso de la puesta en marcha en Hypatia: qué se hizo, por qué, comandos y
resultado. Lo más reciente va al final.

## 1. Datos sintéticos — etapa 1 (humo)

**Objetivo.** Generar el conjunto mínimo para probar que el pipeline corre de punta a punta
(entrenamiento, al menos una validación y evaluación final en test). No se espera que el
modelo aprenda.

**Generador.** `hpc/make_synth_data.py` (solo numpy). Escribe `train.csv`, `val.csv`,
`test.csv` y WAV mono de 16 kHz que cumplen el contrato de datos del código:
columnas `uuid, filename, phq, gad, prompt_vad_trimmed_duration, prompt_vad_cut_duration,
asr_output_id, user`; audios en `<out>/random/audio/<split>/<uuid>.wav`.
Duración 60–61 s porque DAM 1 entrena con `random_adjacent_pair` (2 ventanas de 30 s).
Etiquetas estratificadas para que cada split tenga casos a ambos lados de los cortes de las
métricas (phq > 9, > 14; gad > 4, > 9, > 14).

**Comando (Hypatia, nodo de login).**

```bash
module load anaconda && source activate kintsugi
cd ~/kintsugi-hpc-uniandes
python hpc/make_synth_data.py --mode random --n-train 8 --n-val 4 --n-test 4 \
    --min-dur 60 --max-dur 61 --out-dir <RUTA>/kintsugi_synth/stage1
```

| | train | val | test | total |
|---|---|---|---|---|
| Audios | 8 | 4 | 4 | 16 |
| Esperados tras `filter_metadata` | 8 | 4 | 4 | |

**Resultado.** _Pendiente: pegar la salida del comando (chequeo del contrato y ruta)._
