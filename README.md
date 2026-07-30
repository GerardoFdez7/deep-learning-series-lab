# Lab 2 - Deep Learning (LSTM)

## Estructura del proyecto

```
deep-learning-series-lab/
├── README.md
├── requirements.txt
├── experiments/
│   └── run_avance.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── training.py
├── reports/
│   ├── figures/
│   └── tables/
└── artifacts/
    └── models/
```

## Series usadas en este avance

Para mantener consistencia con el laboratorio anterior se modelan estas 2 series:
1. `total_viajeros`: Turista + Excursionista agregados por mes.
2. `via_aerea`: Total mensual de viajeros por via aerea.

## Modelos incluidos

Se incluyen dos familias LSTM con tuneo por grid:
1. `lstm_simple` (una capa LSTM).
2. `lstm_stacked` (dos capas LSTM).

Cada familia prueba varias combinaciones de:
- `window`
- `units`
- `dropout`
- `learning rate`
- `batch size`

La seleccion del mejor modelo se hace con `val_loss` y luego se reporta desempeno en test (MAE, RMSE, MAPE).

## Como ejecutar

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py experiments/run_avance.py
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python experiments/run_avance.py
```

Opcional para ajustar epocas:

```bash
python experiments/run_avance.py --epochs 60
```

Opcional para correr solo una serie:

```bash
python experiments/run_avance.py --series total_viajeros
```

## Archivos de salida

Al terminar, se generan:
- `reports/tables/tuning_results.csv`
- `reports/tables/best_models_summary.csv`
- `reports/figures/forecast_total_viajeros.png`
- `reports/figures/forecast_via_aerea.png`
- `artifacts/models/best_total_viajeros.keras`
- `artifacts/models/best_via_aerea.keras`
