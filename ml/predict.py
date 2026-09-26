"""Инференс сохранённой модели.

CLI (офлайн, по файлу прогнозных точек):
    python -m ml.predict --model artifacts/model --points ../validate/points.csv --out submission.csv

Программно (для бэкенда): :class:`DelayPredictor` — принимает телеметрию/расписание
в виде DataFrame (как их выдаёт парсер NDTP) и прогнозные точки, возвращает задержки.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .arrivals import detect_arrivals, map_match
from .data import DATA, ROOT, load_points, load_schedule, load_traffic
from .dataset import ARRIVAL_KW, MAPMATCH_KW
from .features import SegmentProfile, build_contexts, build_sequences, build_table
from .models import ModelBundle


class DelayPredictor:
    """Обёртка «модель + контекст» для онлайн/офлайн-прогноза.

    Args:
        model_dir: папка с сохранённым ModelBundle.
    """

    def __init__(self, model_dir: Path):
        self.bundle = ModelBundle.load(Path(model_dir))
        self.ctx = None
        self.prof = None

    def set_context(self, traffic: pd.DataFrame, schedule: pd.DataFrame) -> None:
        """traffic — формат data.load_traffic, schedule — формат data.load_schedule (только план)."""
        sch = detect_arrivals(schedule, traffic, **ARRIVAL_KW)
        tf = map_match(sch, traffic, **MAPMATCH_KW)
        self.ctx = build_contexts(sch, tf)
        self.prof = SegmentProfile(self.ctx)

    def predict(self, points: pd.DataFrame) -> np.ndarray:
        """points: tr_id, T_s, target_stop_id, target_plan_s, cur_dev_s → задержка, сек."""
        X = build_table(points, self.ctx, self.prof)
        S = build_sequences(points, self.ctx)
        return self.bundle.predict(X, S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "artifacts/model"))
    ap.add_argument("--points", default=str(DATA / "validate/points.csv"))
    ap.add_argument("--traffic", default=str(DATA / "validate/traffic.csv"))
    ap.add_argument("--schedule", default=str(DATA / "validate/schedule_plan.csv"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/submission.csv"))
    a = ap.parse_args()
    t0 = time.perf_counter()
    pr = DelayPredictor(Path(a.model))
    pr.set_context(load_traffic(Path(a.traffic)), load_schedule(Path(a.schedule)))
    t1 = time.perf_counter()
    pts = load_points(Path(a.points))
    pred = pr.predict(pts)
    t2 = time.perf_counter()
    pd.DataFrame({"sample_id": pts.sample_id, "prediction": np.round(pred, 1)}).to_csv(a.out, sep=";", index=False)
    print(f"{len(pts)} прогнозов → {a.out}; контекст {t1 - t0:.1f} c, инференс {1000 * (t2 - t1) / len(pts):.2f} мс/точку")


if __name__ == "__main__":
    main()
