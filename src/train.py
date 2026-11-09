from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "laptop_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "laptop_price_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
TARGET = "Price"
REQUIRED_COLUMNS = {
    "Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram",
    "Memory", "Gpu", "OpSys", "Weight", TARGET,
}


def _number(value: object, pattern: str, field: str) -> float:
    match = re.search(pattern, str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse {field}: {value}")
    return float(match.group(1))


def _storage_features(value: object) -> dict[str, float]:
    totals = {"SSD": 0.0, "HDD": 0.0, "Flash Storage": 0.0, "Hybrid": 0.0}
    text = str(value).replace(",", "").upper()
    for part in text.split("+"):
        amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", part)
        if not amount_match:
            continue
        amount = float(amount_match.group(1))
        amount *= 1000 if amount_match.group(2) == "TB" else 1
        for storage_type in totals:
            if storage_type.upper() in part:
                totals[storage_type] += amount
                break
    return totals


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = data.copy()
    frame["Ram"] = frame["Ram"].map(lambda x: _number(x, r"(\d+(?:\.\d+)?)", "RAM"))
    frame["Weight"] = frame["Weight"].map(lambda x: _number(x, r"(\d+(?:\.\d+)?)", "weight"))

    resolution = frame["ScreenResolution"].astype(str).str.extract(
        r"(\d{3,5})\s*x\s*(\d{3,5})"
    )
    frame["X_res"] = pd.to_numeric(resolution[0], errors="coerce")
    frame["Y_res"] = pd.to_numeric(resolution[1], errors="coerce")
    inches = pd.to_numeric(frame["Inches"], errors="coerce").replace(0, np.nan)
    frame["PPI"] = np.sqrt(frame["X_res"] ** 2 + frame["Y_res"] ** 2) / inches
    frame["Touchscreen"] = frame["ScreenResolution"].astype(str).str.contains(
        "touchscreen", case=False, regex=False
    ).astype(int)
    frame["IPS"] = frame["ScreenResolution"].astype(str).str.contains(
        "IPS", case=False, regex=False
    ).astype(int)

    cpu_parts = frame["Cpu"].fillna("Unknown").astype(str).str.split()
    frame["CPU"] = cpu_parts.map(lambda parts: " ".join(parts[:3]))
    frame["CPU family"] = frame["CPU"].map(
        lambda value: value if value.startswith(("Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9"))
        else ("Other Intel" if value.startswith("Intel") else ("AMD" if value.startswith("AMD") else "Other"))
    )
    frame["GPU brand"] = frame["Gpu"].fillna("Unknown").astype(str).str.split().str[0]

    storage = pd.DataFrame(
        frame["Memory"].map(_storage_features).tolist(), index=frame.index
    )
    frame = pd.concat([frame, storage], axis=1)
    frame["OS family"] = frame["OpSys"].fillna("Unknown").map(
        lambda value: "Windows" if str(value).startswith("Windows")
        else ("macOS" if str(value) in {"macOS", "Mac OS X"} else "Linux/Other")
    )

    frame = frame.drop(
        columns=["Inches", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys",
                 "X_res", "Y_res", "CPU"],
        errors="ignore",
    )
    return frame


def build_pipeline(model) -> Pipeline:
    numeric = ["Ram", "Weight", "PPI", "Touchscreen", "IPS", "SSD", "HDD",
               "Flash Storage", "Hybrid"]
    categorical = ["Company", "TypeName", "CPU family", "GPU brand", "OS family"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", model)])


def train(data_path: Path = DATA_PATH, use_mlflow: bool = True) -> dict:
    raw = pd.read_csv(data_path)
    if raw.empty:
        raise ValueError("The laptop dataset is empty.")
    data = engineer_features(raw)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if len(data) < 30:
        raise ValueError("At least 30 valid rows are required for a reliable split and cross-validation.")

    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=250, l2_regularization=1.0,
            random_state=42
        ),
    }

    scores = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        cv_rmse = -cross_val_score(
            pipeline, X_train, y_train, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=1
        )
        scores[name] = {
            "mean_cv_rmse": float(cv_rmse.mean()),
            "std_cv_rmse": float(cv_rmse.std()),
        }

    best_name = min(scores, key=lambda name: scores[name]["mean_cv_rmse"])
    model = build_pipeline(candidates[best_name])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "target": TARGET,
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
        "cross_validation": scores,
        "holdout_metrics": metrics,
        "target_transform": "none",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if use_mlflow:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("laptop-price-prediction")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_param("dataset_rows", len(data))
            mlflow.log_param("split_random_state", 42)
            mlflow.log_metrics(metrics)
            for name, values in scores.items():
                mlflow.log_metric(f"{name}_cv_rmse", values["mean_cv_rmse"])
                mlflow.log_metric(f"{name}_cv_rmse_std", values["std_cv_rmse"])
            mlflow.log_artifact(str(METADATA_PATH))
            mlflow.sklearn.log_model(model, artifact_path="model")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the laptop price regression pipeline.")
    parser.add_argument("--no-mlflow", action="store_true", help="Train without experiment tracking.")
    args = parser.parse_args()
    print(json.dumps(train(use_mlflow=not args.no_mlflow), indent=2))
