"""Загрузка и очистка исходных CSV: телеметрия, расписание, прогнозные точки.

Все времена приводятся к секундам Unix (float64, наивное время как UTC) — так
с ними удобно работать в numpy и так же приходят метки времени в потоке NDTP.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # корень ML-проекта (___ML-MODEL)
# Папка с раздачей хакатона (train/, test/, validate/, labels/): по умолчанию — родительская.
DATA = Path(os.environ.get("MOSTRANS_DATA", ROOT.parent))

# Синтетические ТС в train — зашумлённые копии реальных (корреляция задержек ~0.99),
# поэтому их разметка раскрывает задержки реальных ТС на test/validate. Не используем.
SYNTHETIC_TR_MIN = 9_000_000


def to_sec(s: pd.Series) -> np.ndarray:
    """Строки дат → секунды Unix (float). Пустые значения → NaN."""
    t = pd.to_datetime(s, errors="coerce")
    out = (t - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return out.to_numpy(dtype=np.float64)


def load_traffic(path: Path, drop_synthetic: bool = True) -> pd.DataFrame:
    """Телеметрия: только валидные координаты, без дублей, отсортировано по (tr_id, t).

    Колонки результата: tr_id, t (event_time), rt (receive_time), lon, lat, speed, heading.
    """
    df = pd.read_csv(
        path,
        usecols=["tr_id", "event_time", "receive_time", "location_valid", "lon", "lat", "speed", "heading"],
    )
    if drop_synthetic:
        df = df[df.tr_id < SYNTHETIC_TR_MIN]
    df = df[(df.location_valid.astype(str) == "True") & df.lon.notna() & df.lat.notna()]
    out = pd.DataFrame(
        {
            "tr_id": df.tr_id.to_numpy(np.int64),
            "t": to_sec(df.event_time),
            "rt": to_sec(df.receive_time),
            "lon": df.lon.to_numpy(np.float64),
            "lat": df.lat.to_numpy(np.float64),
            "speed": df.speed.fillna(0).to_numpy(np.float32),
            "heading": df.heading.fillna(0).to_numpy(np.float32),
        }
    )
    out = out.drop_duplicates(["tr_id", "t"]).sort_values(["tr_id", "t"]).reset_index(drop=True)
    return out


def load_schedule(path: Path, drop_synthetic: bool = True) -> pd.DataFrame:
    """Расписание. Колонки: stop_id, tr_id, plan, fact (NaN если нет), lon, lat, address, manual_fill.

    Отсортировано по (tr_id, plan) — это и есть «нитка графика» ТС.
    """
    df = pd.read_csv(path)
    if drop_synthetic:
        df = df[df.tr_id < SYNTHETIC_TR_MIN]
    xy = df.geom.str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    out = pd.DataFrame(
        {
            "stop_id": df.tt_action_item_id.to_numpy(np.int64),
            "tr_id": df.tr_id.to_numpy(np.int64),
            "plan": to_sec(df.time_begin),
            "fact": to_sec(df.time_fact_begin) if "time_fact_begin" in df else np.nan,
            "lon": xy[0].to_numpy(),
            "lat": xy[1].to_numpy(),
            "address": df.building_address.to_numpy(),
            "manual_fill": (df.manual_fill.astype(str) == "True").to_numpy(),
        }
    )
    return out.sort_values(["tr_id", "plan", "stop_id"]).reset_index(drop=True)


def load_points(path: Path) -> pd.DataFrame:
    """Прогнозные точки (labels_* или validate/points). Добавляет T_s, target_plan_s."""
    df = pd.read_csv(path)
    df["T_s"] = to_sec(df["T"])
    df["target_plan_s"] = to_sec(df["target_time_begin"])
    return df
