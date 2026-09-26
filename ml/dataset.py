"""Сборка обучающих/оценочных выборок без утечек.

* Контекст (расписание + телеметрия + GPS-прибытия) строится из файлов нужного периода.
  Для test/validate телеметрия — `test/traffic.csv` (= `validate/traffic.csv`), расписание —
  `validate/schedule_plan.csv` (без факта): фактические времена в модель не попадают.
* Расширение train: моменты T на минутной сетке, цель — первая остановка с планом в
  (T+10, T+15] мин. Берём только остановки, чьё окно-слот размечено как train
  (или не размечено и не соседствует с test/validate), — чтобы не учиться на
  целях test/validate. Метка = fact − plan из `train/schedule.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .arrivals import detect_arrivals, map_match
from .data import DATA, load_points, load_schedule, load_traffic
from .features import WINDOW_HI, WINDOW_LO, SegmentProfile, build_contexts

ARRIVAL_KW = dict(radius=30.0, before=600.0, after=1200.0)
MAPMATCH_KW = dict(w_time=0.05)
SLOT = 300.0


def make_context(traffic_path, schedule_path):
    """Контексты ТС + профиль перегонов для периода."""
    tf = load_traffic(traffic_path)
    sch = load_schedule(schedule_path)
    sch = detect_arrivals(sch, tf, **ARRIVAL_KW)
    tf = map_match(sch, tf, **MAPMATCH_KW)
    ctx = build_contexts(sch, tf)
    return ctx, SegmentProfile(ctx), sch


def slot_of(plan_s: np.ndarray) -> np.ndarray:
    """Слот T (кратен 5 мин), в чьё окно (T+10, T+15] попадает плановое время."""
    return np.ceil((plan_s - WINDOW_HI) / SLOT) * SLOT


def slot_split_map() -> dict[tuple[int, float], str]:
    """(tr_id, T_s) → 'train' / 'test' / 'val' по всем разметкам."""
    m: dict[tuple[int, float], str] = {}
    for name, path in [("train", "labels/labels_train.csv"), ("test", "labels/labels_test.csv"), ("val", "validate/points.csv")]:
        p = load_points(DATA / path)
        for tr, t in zip(p.tr_id, p.T_s):
            m[(int(tr), float(t))] = name
    return m


def augmented_points(sch: pd.DataFrame, allowed: set[str], step: float = 60.0) -> pd.DataFrame:
    """Прогнозные точки на сетке T с шагом step для слотов из allowed ('train', 'test').

    sch — расписание с фактом (train/schedule.csv, реальные ТС).
    """
    split = slot_split_map()
    held = {"train", "test", "val"} - allowed
    rows = []
    for tr, s in sch.groupby("tr_id"):
        plan = s.plan.to_numpy()
        fact = s.fact.to_numpy()
        sid = s.stop_id.to_numpy()
        slots = slot_of(plan)
        lab = np.array([split.get((int(tr), float(x)), "") for x in slots])
        # соседние с отложенными слоты тоже выкидываем, если они не размечены
        held_slots = {float(x) for x, l in zip(slots, lab) if l in held}
        ok = np.array([
            (l in allowed) or (l == "" and not ({x - SLOT, x + SLOT} & held_slots))
            for x, l in zip(slots, lab)
        ])
        grid = np.arange(np.floor((plan.min() - WINDOW_HI) / step) * step, plan.max(), step)
        # первая остановка с плановым временем в (T+10, T+15]
        k = np.searchsorted(plan, grid + WINDOW_LO, side="right")
        valid = (k < len(plan))
        k = np.clip(k, 0, len(plan) - 1)
        valid &= plan[k] <= grid + WINDOW_HI
        # cur_dev_s как у организаторов: последняя остановка с планом ≤ T
        kc = np.searchsorted(plan, grid, side="right") - 1
        valid &= kc >= 0
        for T, kk, kcc in zip(grid[valid], k[valid], kc[valid]):
            if not ok[kk] or not np.isfinite(fact[kk]) or not np.isfinite(fact[kcc]):
                continue
            rows.append((f"{tr}_{int(T)}", int(tr), T, int(sid[kk]), plan[kk], fact[kcc] - plan[kcc],
                         fact[kk] - plan[kk], lab[kk] or "none"))
    df = pd.DataFrame(rows, columns=["sample_id", "tr_id", "T_s", "target_stop_id", "target_plan_s",
                                     "cur_dev_s", "target_delay_s", "slot_split"])
    return df
