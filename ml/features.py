"""Признаки на момент T (строго каузальные: телеметрия с event_time ≤ T, прибытия с known_at ≤ T).

Основной объект — :class:`TrContext`: нитка графика одного ТС + его телеметрия +
прибытия, восстановленные по GPS. :meth:`TrContext.state_at` векторно считает
«состояние ТС» на произвольные моменты времени; на нём строятся и табличные
признаки, и последовательности для нейросети.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arrivals import to_xy

WINDOW_LO, WINDOW_HI = 600.0, 900.0  # окно прогноза (T+10 мин, T+15 мин]
SEQ_LEN, SEQ_STEP = 40, 30.0  # последовательность: 40 шагов по 30 с = 20 минут истории
MOVE_KMH, STOP_KMH = 5.0, 3.0
MM_GOOD_M = 60.0  # точка считается привязанной к нитке, если до перегона < 60 м


def _window_sum(cs: np.ndarray, t: np.ndarray, g: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Сумма и число элементов ряда (через cumsum cs, len = n+1) на интервале (g−w, g]."""
    i1 = np.searchsorted(t, g, side="right")
    i0 = np.searchsorted(t, g - w, side="right")
    return cs[i1] - cs[i0], (i1 - i0).astype(np.float64)


@dataclass
class TrContext:
    """Всё, что известно про одно ТС: расписание, прибытия по GPS, телеметрия."""

    tr_id: int
    plan: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    arr: np.ndarray  # время прибытия по GPS (NaN — не обнаружено)
    stop_ids: np.ndarray
    manual_fill: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    mm_dev: np.ndarray  # задержка по map matching для каждой точки телеметрии
    mm_dist: np.ndarray
    mm_seg: np.ndarray
    stop_index: dict = field(init=False)

    def __post_init__(self) -> None:
        self.stop_index = {int(s): i for i, s in enumerate(self.stop_ids)}
        good = np.isfinite(self.mm_dev) & (self.mm_dist < MM_GOOD_M)
        self.g_t = self.t[good]
        self.g_dev = self.mm_dev[good]
        self.g_seg = self.mm_seg[good]
        self.cs_gdev = np.concatenate([[0.0], np.cumsum(self.g_dev)])
        seg = np.hypot(np.diff(self.sx), np.diff(self.sy))
        self.cumd = np.concatenate([[0.0], np.cumsum(seg)])  # путь по нитке графика, м
        det = np.flatnonzero(np.isfinite(self.arr))
        self.det_k = det
        self.det_t = self.arr[det]  # монотонно (детекция идёт вперёд по времени)
        self.det_dev = self.arr[det] - self.plan[det]
        n = len(self.t)
        step = np.hypot(np.diff(self.x), np.diff(self.y)) if n > 1 else np.zeros(0)
        step = np.where(step < 500, step, 0.0)  # выбросы GPS
        z = lambda a: np.concatenate([[0.0], np.cumsum(a)])
        self.cs_speed = z(self.speed.astype(np.float64))
        self.cs_stop = z((self.speed < STOP_KMH).astype(np.float64))
        self.cs_dist = z(np.concatenate([[0.0], step]))
        self.move_t = self.t[self.speed >= MOVE_KMH]

    # ------------------------------------------------------------------ state
    def state_at(self, g: np.ndarray) -> dict[str, np.ndarray]:
        """Состояние ТС на моменты g (вектор). Использует только данные ≤ g."""
        g = np.asarray(g, dtype=np.float64)
        nstop = len(self.plan)
        out: dict[str, np.ndarray] = {}

        # последнее обнаруженное прибытие на остановку (радиусная детекция)
        j = np.searchsorted(self.det_t, g, side="right") - 1
        has = j >= 0
        jj = np.clip(j, 0, max(len(self.det_t) - 1, 0))
        if len(self.det_t):
            dev_last = np.where(has, self.det_dev[jj], np.nan)
            age_last = np.where(has, g - self.det_t[jj], np.nan)
        else:
            dev_last = np.full(g.shape, np.nan)
            age_last = np.full(g.shape, np.nan)

        # последняя точка, привязанная к нитке (map matching)
        m = np.searchsorted(self.g_t, g, side="right") - 1
        hm = m >= 0
        mc = np.clip(m, 0, max(len(self.g_t) - 1, 0))
        if len(self.g_t):
            dev_mm = np.where(hm, self.g_dev[mc], np.nan)
            age_mm = np.where(hm, g - self.g_t[mc], np.nan)
            seg_mm = np.where(hm, self.g_seg[mc], -1)
        else:
            dev_mm = age_mm = np.full(g.shape, np.nan)
            seg_mm = np.full(g.shape, -1)
        fresh = hm & (age_mm < 900)
        # позиция на нитке: перегон из привязки, иначе — по плану
        k_plan = np.searchsorted(self.plan, g, side="right") - 1
        k_cur = np.clip(np.where(fresh, seg_mm, k_plan), 0, nstop - 1)
        n_next = np.clip(k_cur + 1, 0, nstop - 1)

        # нижняя граница опоздания: следующую остановку ещё не прошли
        lb = g - self.plan[n_next]
        # текущая задержка: привязка + «застаивание» с момента последней привязанной точки,
        # но не меньше нижней границы
        dev_now = np.where(fresh, np.maximum(dev_mm, lb), np.where(has, np.maximum(dev_last, lb), 0.0))
        for w in (120.0, 300.0):
            s, c = _window_sum(self.cs_gdev, self.g_t, g, w)
            out[f"dev_mm_mean_{int(w)}"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)

        # позиция / свежесть телеметрии
        ip = np.searchsorted(self.t, g, side="right") - 1
        hp = ip >= 0
        ipc = np.clip(ip, 0, max(len(self.t) - 1, 0))
        if len(self.t):
            px, py = self.x[ipc], self.y[ipc]
            age_pos = np.where(hp, g - self.t[ipc], np.nan)
            cur_speed = np.where(hp, self.speed[ipc], np.nan)
        else:
            px = py = age_pos = cur_speed = np.full(g.shape, np.nan)
        d2 = np.hypot(px - self.sx[n_next], py - self.sy[n_next])

        out.update(
            k_last=k_cur, n_next=n_next, has_arr=has.astype(np.float32), dev_last=dev_last,
            age_last=age_last, dev_lb=lb, dev_now=dev_now, dev_interp=dev_mm, age_mm=age_mm,
            fresh=fresh.astype(np.float32), px=px, py=py, d_next=d2, age_pos=age_pos, cur_speed=cur_speed,
        )

        # скорость / простой / пройденный путь в окнах
        for w in (120.0, 300.0, 600.0):
            s, c = _window_sum(self.cs_speed, self.t, g, w)
            st, _ = _window_sum(self.cs_stop, self.t, g, w)
            d, _ = _window_sum(self.cs_dist, self.t, g, w)
            tag = int(w)
            out[f"spd_mean_{tag}"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            out[f"stop_frac_{tag}"] = np.where(c > 0, st / np.maximum(c, 1), np.nan)
            out[f"dist_{tag}"] = d
            out[f"npts_{tag}"] = c
        im = np.searchsorted(self.move_t, g, side="right") - 1
        out["dwell"] = np.where(im >= 0, g - self.move_t[np.clip(im, 0, None)] if len(self.move_t) else np.nan, np.nan)
        return out

    # ---------------------------------------------------- history of arrivals
    def dev_slope(self, g: np.ndarray, w: float = 900.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Наклон (сек задержки / мин) и среднее задержки по прибытиям в (g−w, g], их число."""
        slope = np.full(len(g), np.nan)
        mean = np.full(len(g), np.nan)
        cnt = np.zeros(len(g))
        i1 = np.searchsorted(self.det_t, g, side="right")
        i0 = np.searchsorted(self.det_t, g - w, side="right")
        for i, (a, b) in enumerate(zip(i0, i1)):
            if b - a >= 1:
                mean[i] = self.det_dev[a:b].mean()
                cnt[i] = b - a
            if b - a >= 3:
                tt = (self.det_t[a:b] - self.det_t[a:b].mean()) / 60.0
                dd = self.det_dev[a:b]
                den = (tt * tt).sum()
                slope[i] = (tt * (dd - dd.mean())).sum() / den if den > 0 else 0.0
        return slope, mean, cnt


def build_contexts(schedule: pd.DataFrame, traffic: pd.DataFrame) -> dict[int, TrContext]:
    """schedule должен содержать arr_gps (arrivals.detect_arrivals), traffic — mm_* (arrivals.map_match)."""
    ctx: dict[int, TrContext] = {}
    tgroups = dict(tuple(traffic.groupby("tr_id")))
    for tr, s in schedule.groupby("tr_id"):
        tf = tgroups.get(tr, traffic.iloc[:0])
        sx, sy = to_xy(s.lon.to_numpy(), s.lat.to_numpy())
        px, py = to_xy(tf.lon.to_numpy(), tf.lat.to_numpy())
        ctx[int(tr)] = TrContext(
            tr_id=int(tr), plan=s.plan.to_numpy(), sx=sx, sy=sy, arr=s.arr_gps.to_numpy(),
            stop_ids=s.stop_id.to_numpy(), manual_fill=s.manual_fill.to_numpy(),
            t=tf.t.to_numpy(), x=px, y=py, speed=tf.speed.to_numpy(),
            mm_dev=tf.mm_dev.to_numpy(), mm_dist=tf.mm_dist.to_numpy(), mm_seg=tf.mm_seg.to_numpy(),
        )
    return ctx


# ---------------------------------------------------------------- segments
class SegmentProfile:
    """Каузальный профиль перегонов: насколько фактический ход отличается от планового.

    Наблюдения берутся по GPS-прибытиям всех ТС; для запроса на момент T используются
    только наблюдения, известные к T.
    """

    def __init__(self, contexts: dict[int, TrContext]):
        obs: dict[tuple, list] = {}
        for c in contexts.values():
            keys = self.keys(c)
            for k in range(1, len(c.plan)):
                if np.isfinite(c.arr[k]) and np.isfinite(c.arr[k - 1]):
                    run_dev = (c.arr[k] - c.arr[k - 1]) - (c.plan[k] - c.plan[k - 1])
                    if abs(run_dev) < 900:
                        obs.setdefault(keys[k], []).append((c.arr[k], run_dev))
        self.t: dict[tuple, np.ndarray] = {}
        self.cs: dict[tuple, np.ndarray] = {}
        for key, v in obs.items():
            v.sort()
            a = np.array(v)
            self.t[key] = a[:, 0]
            self.cs[key] = np.concatenate([[0.0], np.cumsum(a[:, 1])])

    @staticmethod
    def keys(c: TrContext) -> list[tuple]:
        """Ключ перегона k-1 → k: округлённые координаты обеих остановок."""
        r = lambda v: int(round(v / 20.0))
        ks = [None]
        for k in range(1, len(c.plan)):
            ks.append((r(c.sx[k - 1]), r(c.sy[k - 1]), r(c.sx[k]), r(c.sy[k])))
        return ks

    def ahead(self, c: TrContext, keys: list, k_from: int, k_to: int, g: float) -> tuple[float, float, float]:
        """Сумма средних отклонений хода по перегонам (k_from, k_to], число покрытых, всего."""
        tot, cov, n = 0.0, 0.0, 0.0
        for k in range(max(k_from + 1, 1), k_to + 1):
            n += 1
            key = keys[k]
            if key not in self.t:
                continue
            i = np.searchsorted(self.t[key], g, side="right")
            if i > 0:
                tot += self.cs[key][i] / i
                cov += 1
        return tot, cov, n


# ---------------------------------------------------------------- tabular
FEATURES = [
    "cur_dev", "horizon", "dev_last", "age_last", "dev_lb", "dev_now", "dev_interp", "has_arr",
    "dev_now_m5", "dev_now_m10", "dev_trend5", "slope15", "mean_dev15", "n_arr15", "slope30", "mean_dev30",
    "stops_ahead", "plan_to_target", "dist_to_target", "max_gap_ahead", "slack_ahead", "layover",
    "d_next", "age_pos", "cur_speed", "dwell",
    "spd_mean_120", "spd_mean_300", "spd_mean_600", "stop_frac_120", "stop_frac_300", "stop_frac_600",
    "dist_120", "dist_300", "dist_600", "npts_300",
    "eta_dev", "plan_speed_ahead", "seg_prof_ahead", "seg_prof_cov",
    "cur_minus_now", "age_mm", "fresh", "dev_mm_mean_120", "dev_mm_mean_300",
    "cur_dev_is0", "det_cov30", "mm_cov30", "stops_ahead_plan",
    "manual_fill",
]
# manual_fill целевой остановки — флаг «факт внесён вручную» из schedule_plan. Он есть во
# входах validate, но описывает способ записи факта (86% таких фактов = плану), поэтому
# по умолчанию модель его НЕ использует (см. train.py --use-manual-fill).
OPTIONAL_FEATURES = {"manual_fill"}


def build_table(points: pd.DataFrame, contexts: dict[int, TrContext], profile: SegmentProfile | None) -> pd.DataFrame:
    """Табличные признаки для прогнозных точек (tr_id, T_s, target_stop_id, target_plan_s, cur_dev_s)."""
    rows = []
    for tr, p in points.groupby("tr_id", sort=False):
        c = contexts[int(tr)]
        keys = SegmentProfile.keys(c) if profile is not None else None
        g = p.T_s.to_numpy()
        st = c.state_at(g)
        st5 = c.state_at(g - 300)
        st10 = c.state_at(g - 600)
        sl15, md15, n15 = c.dev_slope(g, 900)
        sl30, md30, _ = c.dev_slope(g, 1800)
        tgt = np.array([c.stop_index.get(int(s), -1) for s in p.target_stop_id])
        for i in range(len(p)):
            ti = tgt[i]
            if ti < 0:  # целевая остановка не найдена в нитке — ставим по плановому времени
                ti = int(np.clip(np.searchsorted(c.plan, p.target_plan_s.iat[i]), 0, len(c.plan) - 1))
            nn = int(st["n_next"][i])
            kf = min(nn, ti)
            gaps = np.diff(c.plan[kf : ti + 1]) if ti > kf else np.zeros(0)
            dist_t = c.cumd[ti] - c.cumd[kf] + (st["d_next"][i] if np.isfinite(st["d_next"][i]) else 0.0)
            plan_to = c.plan[ti] - max(g[i], c.plan[kf]) if ti >= kf else 0.0
            spd = st["spd_mean_600"][i]
            mv = st["dist_600"][i] / 600.0  # м/с по фактическому пути
            eta = g[i] + dist_t / max(mv, 1.0) - c.plan[ti] if np.isfinite(mv) else np.nan
            if profile is not None:
                sp, cov, nseg = profile.ahead(c, keys, max(int(st["k_last"][i]), 0), ti, g[i])
            else:
                sp, cov, nseg = np.nan, np.nan, np.nan
            r = {
                "cur_dev": p.cur_dev_s.iat[i],
                "horizon": p.target_plan_s.iat[i] - g[i],
                "dev_now_m5": st5["dev_now"][i],
                "dev_now_m10": st10["dev_now"][i],
                "dev_trend5": st["dev_now"][i] - st5["dev_now"][i],
                "slope15": sl15[i], "mean_dev15": md15[i], "n_arr15": n15[i],
                "slope30": sl30[i], "mean_dev30": md30[i],
                "stops_ahead": ti - int(st["k_last"][i]),
                "plan_to_target": plan_to,
                "dist_to_target": dist_t,
                "max_gap_ahead": gaps.max() if gaps.size else 0.0,
                "slack_ahead": np.clip(gaps - 120, 0, None).sum() if gaps.size else 0.0,
                "layover": float(gaps.size and gaps.max() > 300),
                "eta_dev": eta,
                "plan_speed_ahead": dist_t / max(plan_to, 60.0),
                "seg_prof_ahead": sp,
                "seg_prof_cov": cov / nseg if nseg else np.nan,
                "manual_fill": float(c.manual_fill[ti]),
            }
            for k in ("dev_last", "age_last", "dev_lb", "dev_now", "dev_interp", "has_arr", "d_next",
                      "age_pos", "cur_speed", "dwell", "spd_mean_120", "spd_mean_300", "spd_mean_600",
                      "stop_frac_120", "stop_frac_300", "stop_frac_600", "dist_120", "dist_300",
                      "dist_600", "npts_300", "age_mm", "fresh", "dev_mm_mean_120", "dev_mm_mean_300"):
                r[k] = st[k][i]
            r["cur_minus_now"] = r["cur_dev"] - r["dev_now"]
            r["cur_dev_is0"] = float(r["cur_dev"] == 0)
            # покрытие GPS за 30 минут: доля плановых остановок с обнаруженным прибытием
            # и доля времени с привязанными к нитке точками
            k0 = np.searchsorted(c.plan, g[i] - 1800, side="right")
            k1 = np.searchsorted(c.plan, g[i], side="right")
            if k1 > k0:
                a = c.arr[k0:k1]
                r["det_cov30"] = float(np.mean(np.isfinite(a) & (a <= g[i])))
            else:
                r["det_cov30"] = np.nan
            j0 = np.searchsorted(c.g_t, g[i] - 1800, side="right")
            j1 = np.searchsorted(c.g_t, g[i], side="right")
            r["mm_cov30"] = (j1 - j0) / 180.0  # ~1 точка / 10 с
            r["stops_ahead_plan"] = ti - (np.searchsorted(c.plan, g[i], side="right") - 1)
            r["_idx"] = p.index[i]
            rows.append(r)
    df = pd.DataFrame(rows).set_index("_idx").loc[points.index]
    return df[FEATURES].astype(np.float32)


# ---------------------------------------------------------------- sequences
SEQ_CHANNELS = ["dev_now", "dev_interp", "cur_speed", "spd_mean_120", "stop_frac_120", "d_next", "age_pos", "dist_120"]


def build_sequences(points: pd.DataFrame, contexts: dict[int, TrContext]) -> np.ndarray:
    """Последовательности состояния ТС за 20 минут до T: [N, SEQ_LEN, len(SEQ_CHANNELS)]."""
    out = np.zeros((len(points), SEQ_LEN, len(SEQ_CHANNELS)), dtype=np.float32)
    offs = -SEQ_STEP * np.arange(SEQ_LEN - 1, -1, -1)
    pos = {ix: i for i, ix in enumerate(points.index)}
    for tr, p in points.groupby("tr_id", sort=False):
        c = contexts[int(tr)]
        g = (p.T_s.to_numpy()[:, None] + offs[None, :]).ravel()
        st = c.state_at(g)
        arr = np.stack([st[ch] for ch in SEQ_CHANNELS], axis=-1).reshape(len(p), SEQ_LEN, -1)
        for i, ix in enumerate(p.index):
            out[pos[ix]] = arr[i]
    return out
