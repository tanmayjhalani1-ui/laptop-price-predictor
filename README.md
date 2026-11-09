<div align="center">
  <img src="assets/portfolio-banner.svg" alt="Laptop Price Prediction — Tanmay Jhalani" width="100%" />
</div>

<div align="center">
  <strong>Designed and engineered by Tanmay Jhalani</strong> · <a href="https://github.com/tanmayjhalani1-ui">GitHub Profile</a>
</div>

---

# Laptop Price Predictor

An end-to-end laptop price regression project built around the repository's original dataset and notebook. The project provides a reusable scikit-learn training pipeline, cross-validated model selection, MLflow experiment tracking, a Streamlit application, a lightweight FastAPI service, and Docker support.

## Project structure

```text
.
├── api/
│   └── main.py
├── artifacts/                    # Generated model and metadata (git-ignored)
├── src/
│   ├── inference.py
│   └── train.py
├── .github/workflows/ci.yml
├── app.py
├── docker-compose.yml
├── Dockerfile
├── laptop-price-predictor.ipynb  # Original notebook, retained
├── laptop_data.csv               # Original dataset, retained
├── laptopprice.py                # Original application, retained
└── requirements.txt
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train and track experiments

```bash
python -m src.train
```

Training validates the source schema, engineers laptop specifications, compares Random Forest, Extra Trees, and HistGradientBoosting regressors using shuffled five-fold cross-validation, selects the lowest mean CV RMSE, and evaluates the selected pipeline on a held-out test split. Preprocessing is fitted inside each cross-validation fold to avoid leakage.

The trained pipeline and evaluation metadata are saved under `artifacts/`. MLflow runs are stored locally in `mlruns/`.

```bash
mlflow ui --backend-store-uri ./mlruns
```

To train without experiment tracking:

```bash
python -m src.train --no-mlflow
```

## Run the application

```bash
streamlit run app.py
```

The Streamlit app includes laptop price estimation, dataset exploration, and model evaluation. The first training run must be completed before launching the app.

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — service health
- `POST /predict` — estimate price from laptop specifications
- Interactive API documentation: `/docs`

The prediction request uses the source dataset's raw specification fields: `Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, and `Weight`.

## Docker

```bash
docker compose up --build
```

The Streamlit app is exposed on port 8501 and the API on port 8000.

## Methodology

- RAM and weight are parsed into numeric features.
- Screen resolution and diagonal size are used to derive pixel density; touchscreen and IPS indicators are extracted.
- CPU family, GPU brand, operating-system family, and storage capacities by storage type are derived from the raw columns.
- Numeric features use median imputation and standard scaling.
- Categorical features use most-frequent imputation and one-hot encoding with unknown-category handling.
- The preprocessing and estimator are kept in one pipeline so training and inference use the same transformations.

## Dataset and limitations

The committed `laptop_data.csv` is retained as the source of truth. Its prices and specifications represent the dataset's collection period and coverage, not today's retail market. Predictions are estimates in the dataset's original price units; no currency conversion or live-price lookup is performed. Results may be less reliable for specifications or brands that are poorly represented in the training data.

## Original work

The original `laptop-price-predictor.ipynb`, `laptopprice.py`, and `laptop_data.csv` remain in the repository alongside the reproducible application and API implementation.
