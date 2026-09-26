"""Анализ (только train): насколько таргет объясняется «идеальной» текущей задержкой.

Использует факты расписания train — ТОЛЬКО для анализа потолка, не для признаков.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.data import DATA, ROOT, load_schedule  # noqa: E402
from ml.dataset import augmented_points  # noqa: E402

sch = load_schedule(DATA / "train/schedule.csv")
aug = augmented_points(sch, {"train"})
y = aug.target_delay_s.to_numpy()
cur = aug.cur_dev_s.to_numpy()
true_now = np.full(len(aug), np.nan)
lag_now = np.full(len(aug), np.nan)
by = {tr: s for tr, s in sch.groupby("tr_id")}
for i, (tr, T) in enumerate(zip(aug.tr_id, aug.T_s)):
    s = by[tr]
    f = s.fact.to_numpy()
    p = s.plan.to_numpy()
    order = np.argsort(f)
    j = np.searchsorted(f[order], T, side="right") - 1
    if j >= 0:
        k = order[j]
        true_now[i] = f[k] - p[k]
        lag_now[i] = T - f[k]
ok = np.isfinite(true_now)
print("n", len(y), "ok", ok.mean().round(3))
print("MAE zero", np.abs(y).mean().round(1), "cur_dev", np.abs(y - cur).mean().round(1),
      "true_now", np.abs(y - true_now)[ok].mean().round(1))
# with lower bound using next plan
print("corr y~cur", np.corrcoef(y, cur)[0, 1].round(3), "y~true_now", np.corrcoef(y[ok], true_now[ok])[0, 1].round(3))
# optimal shrink a*x
for name, x in (("cur", cur), ("true_now", np.where(ok, true_now, cur))):
    best = min(((np.abs(y - a * x - b).mean(), a, b) for a in np.linspace(0.3, 1.2, 19) for b in range(-40, 41, 5)))
    print(name, "best a*x+b MAE %.1f a=%.2f b=%d" % best)
print("cur - true_now diff quantiles", np.nanquantile(cur - true_now, [.05, .25, .5, .75, .95]).round(1))
print("target abs quantiles", np.quantile(np.abs(y), [.25, .5, .75, .9, .99]).round(0))
