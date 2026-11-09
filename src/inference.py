from pathlib import Path

import joblib
import pandas as pd

from src.train import engineer_features, MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python -m src.train' before making predictions."
        )
    return joblib.load(model_path)


def predict_price(model, laptop: dict) -> float:
    raw = pd.DataFrame([laptop])
    features = engineer_features(raw)
    return float(model.predict(features)[0])
