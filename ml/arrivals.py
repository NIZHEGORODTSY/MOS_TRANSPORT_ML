"""Восстановление фактических прибытий на остановки по GPS (упрощённый map matching).

Идея: нитка графика ТС — упорядоченная по плану последовательность остановок.
Для очередной остановки ищем первую точку телеметрии, попавшую в радиус R от
остановки, в окне [plan − before, plan + after] и не раньше предыдущего прибытия.
Детекция каузальна: прибытие становится «известным» в момент получения этой точки,
поэтому на момент T можно пользоваться только прибытиями с known_at ≤ T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAT0 = 55.75
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.deg2rad(LAT0))


def to_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Локальная равнопромежуточная проекция (метры) — достаточно точна в пределах Москвы."""
    return (np.asarray(lon) - 37.6) * M_PER_DEG_LON, (np.asarray(lat) - LAT0) * M_PER_DEG_LAT


def detect_arrivals_one(
    plan: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    t: np.ndarray,
    rt: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    radius: float = 60.0,
    before: float = 600.0,
    after: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Прибытия для одного ТС.

    Args:
        plan: плановые времена остановок (отсортированы).
        sx, sy: координаты остановок, м.
        t, rt: event_time / receive_time точек телеметрии (отсортированы по t).
        px, py: координаты точек, м.

    Returns:
        (arr, known_at): время прибытия и момент, когда оно стало известно; NaN — не найдено.
    """
    n = len(plan)
    arr = np.full(n, np.nan)
    known = np.full(n, np.nan)
    last = -np.inf
    for k in range(n):
        lo = max(plan[k] - before, last)
        i0 = np.searchsorted(t, lo, side="left")
        i1 = np.searchsorted(t, plan[k] + after, side="right")
        if i1 <= i0:
            continue
        d2 = (px[i0:i1] - sx[k]) ** 2 + (py[i0:i1] - sy[k]) ** 2
        hit = np.flatnonzero(d2 < radius * radius)
        if hit.size == 0:
            continue
        j = i0 + hit[0]
        arr[k] = t[j]
        known[k] = max(t[j], rt[j]) if np.isfinite(rt[j]) else t[j]
        last = t[j]
    return arr, known


def map_match_one(
    plan: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    t: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    back: float = 1500.0,
    ahead: float = 900.0,
    w_time: float = 0.15,
    w_head: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Непрерывный каузальный map matching точек телеметрии на перегоны нитки графика.

    Для точки в момент t рассматриваются перегоны k→k+1, плановый интервал которых
    пересекает [t − back, t + ahead] (т.е. задержка от −ahead до +back). Точка
    проецируется на отрезок между остановками; стоимость кандидата =
    перпендикулярное расстояние (м) + w_time·|задержка − предыдущая задержка| +
    штраф w_head за курс против направления перегона. Предыдущая задержка берётся
    только из прошлого, поэтому результат каузален.

    Returns:
        dev: задержка в момент t относительно плановой позиции (с), NaN — нет кандидатов.
        dist: расстояние до выбранного перегона (м) — качество привязки.
        seg: индекс k выбранного перегона k→k+1.
    """
    n = len(t)
    dev = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    seg = np.full(n, -1, dtype=np.int64)
    if len(plan) < 2 or n == 0:
        return dev, dist, seg
    ax, ay, bx, by = sx[:-1], sy[:-1], sx[1:], sy[1:]
    vx, vy = bx - ax, by - ay
    L2 = np.maximum(vx * vx + vy * vy, 1.0)
    bearing = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0  # 0 = север, по часовой
    p0, p1 = plan[:-1], plan[1:]
    # Перегоны с длинной плановой паузой (отстой на конечной, перерыв): по плану ТС стоит
    # в начале перегона и отправляется за «время хода» до p1 (скорость ~6 м/с, не меньше минуты).
    run = np.maximum(60.0, np.sqrt(L2) / 6.0)
    layover = (p1 - p0) > 240.0
    p0 = np.where(layover, np.maximum(p0, p1 - run), p0)
    prev = 0.0
    prev_t = -np.inf
    for i in range(n):
        ti = t[i]
        lo = np.searchsorted(p1, ti - back, side="left")
        hi = np.searchsorted(p0, ti + ahead, side="right")
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        f = np.clip(((px[i] - ax[sl]) * vx[sl] + (py[i] - ay[sl]) * vy[sl]) / L2[sl], 0.0, 1.0)
        qx, qy = ax[sl] + f * vx[sl], ay[sl] + f * vy[sl]
        d = np.hypot(px[i] - qx, py[i] - qy)
        dv = ti - (p0[sl] + f * (p1[sl] - p0[sl]))
        # ожидание на конечной до планового отправления — это «по графику»
        dv = np.where(layover[sl] & (f < 0.1) & (dv < 0), 0.0, dv)
        # предыдущая оценка «устаревает» за ~10 минут
        wt = w_time if ti - prev_t < 600 else w_time * 0.2
        cost = d + wt * np.abs(dv - prev)
        if speed[i] >= 5.0:
            ang = np.abs((heading[i] - bearing[sl] + 180.0) % 360.0 - 180.0)
            cost = cost + w_head * ((ang > 70.0) & (L2[sl] > 2500.0))
        j = int(np.argmin(cost))
        dev[i], dist[i], seg[i] = dv[j], d[j], lo + j
        if d[j] < 80.0:
            prev, prev_t = dv[j], ti
    return dev, dist, seg


def map_match(schedule: pd.DataFrame, traffic: pd.DataFrame, **kw) -> pd.DataFrame:
    """Добавляет к телеметрии колонки mm_dev, mm_dist, mm_seg (индекс в нитке своего ТС)."""
    tf = traffic.copy()
    tf["mm_dev"] = np.nan
    tf["mm_dist"] = np.nan
    tf["mm_seg"] = -1
    sgroups = dict(tuple(schedule.groupby("tr_id")))
    for tr, idx in tf.groupby("tr_id").groups.items():
        s = sgroups.get(tr)
        if s is None:
            continue
        sx, sy = to_xy(s.lon.to_numpy(), s.lat.to_numpy())
        p = tf.loc[idx]
        px, py = to_xy(p.lon.to_numpy(), p.lat.to_numpy())
        dev, dist, seg = map_match_one(
            s.plan.to_numpy(), sx, sy, p.t.to_numpy(), px, py,
            p.heading.to_numpy(), p.speed.to_numpy(), **kw,
        )
        tf.loc[idx, "mm_dev"] = dev
        tf.loc[idx, "mm_dist"] = dist
        tf.loc[idx, "mm_seg"] = seg
    return tf


def detect_arrivals(schedule: pd.DataFrame, traffic: pd.DataFrame, **kw) -> pd.DataFrame:
    """Добавляет к расписанию колонки arr_gps и known_at (по всем ТС)."""
    sch = schedule.copy()
    sch["arr_gps"] = np.nan
    sch["known_at"] = np.nan
    sx, sy = to_xy(sch.lon.to_numpy(), sch.lat.to_numpy())
    sch["x"], sch["y"] = sx, sy
    groups = dict(tuple(traffic.groupby("tr_id")))
    for tr, idx in sch.groupby("tr_id").groups.items():
        tf = groups.get(tr)
        if tf is None:
            continue
        px, py = to_xy(tf.lon.to_numpy(), tf.lat.to_numpy())
        s = sch.loc[idx]
        arr, known = detect_arrivals_one(
            s.plan.to_numpy(), s.x.to_numpy(), s.y.to_numpy(),
            tf.t.to_numpy(), tf.rt.to_numpy(), px, py, **kw,
        )
        sch.loc[idx, "arr_gps"] = arr
        sch.loc[idx, "known_at"] = known
    return sch
