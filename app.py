"""
FastAPI-сервер для предсказания задержек автобусов.

Запуск:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predict import DelayPredictor


# ============================================================
# КОНСТАНТЫ
# ============================================================
MODEL_DIR = Path(__file__).parent / "artifacts" / "model"


# ============================================================
# СХЕМЫ ЗАПРОСОВ / ОТВЕТОВ
# ============================================================
class TrafficRow(BaseModel):
    tr_id: Any
    event_time: str
    receive_time: Optional[str] = None
    lon: float
    lat: float
    speed: Optional[float] = 0.0
    heading: Optional[float] = 0.0
    location_valid: Optional[Any] = True


class ScheduleRow(BaseModel):
    tt_action_item_id: Any
    tr_id: Any
    time_begin: str
    geom: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    building_address: Optional[str] = ""
    manual_fill: Optional[Any] = False


class PointRow(BaseModel):
    sample_id: str
    tr_id: Any
    T: str
    target_stop_id: Any
    target_time_begin: str
    cur_dev_s: float


class SetContextRequest(BaseModel):
    traffic: list[TrafficRow] = Field(..., min_length=1)
    schedule: list[ScheduleRow] = Field(..., min_length=1)


class SetContextResponse(BaseModel):
    n_vehicles: int
    elapsed_sec: float


class PredictRequest(BaseModel):
    points: list[PointRow] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    sample_id: list[str]
    prediction: list[float]


class HealthResponse(BaseModel):
    status: str
    has_context: bool
    n_vehicles: int
    model_dir: str


# ============================================================
# КОНВЕРТЕРЫ JSON -> DataFrame
# ============================================================
def _to_sec(s: pd.Series) -> np.ndarray:
    t = pd.to_datetime(s, errors="coerce")
    return (t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(np.float64)


def traffic_from_rows(rows: list[TrafficRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["tr_id"] = pd.to_numeric(df["tr_id"], errors="coerce")
    df = df.dropna(subset=["tr_id"])
    df["tr_id"] = df.tr_id.astype(np.int64)

    if "location_valid" in df.columns:
        lv = df.location_valid.astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[lv]
    df = df.dropna(subset=["event_time", "lon", "lat"])

    out = pd.DataFrame({
        "tr_id":   df.tr_id.to_numpy(np.int64),
        "t":       _to_sec(df.event_time),
        "rt":      _to_sec(df.receive_time) if "receive_time" in df.columns else _to_sec(df.event_time),
        "lon":     pd.to_numeric(df.lon, errors="coerce").to_numpy(np.float64),
        "lat":     pd.to_numeric(df.lat, errors="coerce").to_numpy(np.float64),
        "speed":   pd.to_numeric(df.speed, errors="coerce").fillna(0).to_numpy(np.float32),
        "heading": pd.to_numeric(df.heading, errors="coerce").fillna(0).to_numpy(np.float32),
    })
    out = out.dropna(subset=["t", "lon", "lat"])
    return out.drop_duplicates(["tr_id", "t"]).sort_values(["tr_id", "t"]).reset_index(drop=True)


def schedule_from_rows(rows: list[ScheduleRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["tr_id"] = pd.to_numeric(df["tr_id"], errors="coerce")
    df = df.dropna(subset=["tr_id"])
    df["tr_id"] = df.tr_id.astype(np.int64)

    if "geom" in df.columns and df.geom.notna().any():
        xy = df.geom.astype(str).str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
        lon, lat = xy[0].to_numpy(), xy[1].to_numpy()
    else:
        lon = pd.to_numeric(df.lon, errors="coerce").to_numpy()
        lat = pd.to_numeric(df.lat, errors="coerce").to_numpy()

    mf = (df.manual_fill.astype(str).str.lower().isin(["true", "1"]).to_numpy()
          if "manual_fill" in df.columns else np.zeros(len(df), bool))

    out = pd.DataFrame({
        "stop_id":     pd.to_numeric(df.tt_action_item_id, errors="coerce").to_numpy(np.int64),
        "tr_id":       df.tr_id.to_numpy(np.int64),
        "plan":        _to_sec(df.time_begin),
        "fact":        np.nan,
        "lon":         lon,
        "lat":         lat,
        "address":     df.building_address.astype(str).to_numpy() if "building_address" in df.columns else "",
        "manual_fill": mf,
    })
    out = out.dropna(subset=["plan", "lon", "lat"])
    return out.sort_values(["tr_id", "plan", "stop_id"]).reset_index(drop=True)


def points_from_rows(rows: list[PointRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["T_s"]           = _to_sec(df["T"])
    df["target_plan_s"] = _to_sec(df["target_time_begin"])
    df["tr_id"]         = pd.to_numeric(df.tr_id, errors="coerce").astype(np.int64)
    return df


# ============================================================
# СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# ============================================================
class AppState:
    def __init__(self):
        self.pr: Optional[DelayPredictor] = None
        self.lock = threading.Lock()
        self.n_vehicles: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Загружаю модель из {MODEL_DIR}")
    state.pr = DelayPredictor(MODEL_DIR)
    print("[startup] Модель готова")
    yield
    print("[shutdown] Остановлено")


app = FastAPI(title="Bus Delay Predictor", lifespan=lifespan)


# ============================================================
# ЭНДПОИНТЫ
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        has_context=state.n_vehicles > 0,
        n_vehicles=state.n_vehicles,
        model_dir=str(MODEL_DIR),
    )


@app.post("/set_context", response_model=SetContextResponse)
def set_context(req: SetContextRequest):
    if state.pr is None:
        raise HTTPException(503, "Модель не загружена")

    t0 = time.time()

    try:
        traffic_df  = traffic_from_rows(req.traffic)
        schedule_df = schedule_from_rows(req.schedule)
    except Exception as e:
        raise HTTPException(400, f"Ошибка парсинга: {e}")

    if len(traffic_df) == 0:
        raise HTTPException(400, "traffic пуст после чистки")
    if len(schedule_df) == 0:
        raise HTTPException(400, "schedule пуст после чистки")

    with state.lock:
        try:
            state.pr.set_context(traffic_df, schedule_df)
        except Exception as e:
            raise HTTPException(500, f"Ошибка построения контекста: {e}")
        state.n_vehicles = len(state.pr.ctx)

    return SetContextResponse(
        n_vehicles=state.n_vehicles,
        elapsed_sec=round(time.time() - t0, 2),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state.pr is None:
        raise HTTPException(503, "Модель не загружена")
    if state.n_vehicles == 0:
        raise HTTPException(409, "Контекст не построен. Сначала вызови /set_context")

    try:
        points_df = points_from_rows(req.points)
    except Exception as e:
        raise HTTPException(400, f"Ошибка парсинга points: {e}")

    if len(points_df) == 0:
        raise HTTPException(400, "points пуст")

    with state.lock:
        try:
            pred = state.pr.predict(points_df)
        except Exception as e:
            raise HTTPException(500, f"Ошибка предсказания: {e}")

    return PredictResponse(
        sample_id=points_df.sample_id.tolist(),
        prediction=[round(float(x), 1) for x in pred],
    )
