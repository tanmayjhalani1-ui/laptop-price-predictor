from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_price

app = FastAPI(title="Laptop Price Prediction API", version="1.0.0")


class LaptopInput(BaseModel):
    Company: str
    TypeName: str
    Inches: float = Field(gt=0, le=30)
    ScreenResolution: str
    Cpu: str
    Ram: str
    Memory: str
    Gpu: str
    OpSys: str
    Weight: str


@lru_cache
def get_model():
    return load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: LaptopInput):
    try:
        estimate = predict_price(get_model(), payload.model_dump())
        return {"estimated_price": round(estimate, 2), "currency": "dataset_units"}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
