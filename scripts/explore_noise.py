"""Анализ (только train): гладкость ряда задержек вдоль нитки и связь с таргетом."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.data import DATA, load_schedule  # noqa: E402

sch = load_schedule(DATA / "train/schedule.csv")
d_all, lags = [], {1: [], 3: [], 10: []}
for tr, s in sch.groupby("tr_id"):
    d = (s.fact - s.plan).to_numpy()
    d_all.append(d)
    for L in lags:
        lags[L].append(np.abs(d[L:] - d[:-L]))
for L, v in lags.items():
    v = np.concatenate(v)
    print(f"|d[k+{L}]-d[k]| mean {v.mean():.1f} median {np.median(v):.1f}")
d = np.concatenate(d_all)
print("delay quantiles", np.quantile(d, [.01, .1, .25, .5, .75, .9, .99]).round(0))
print("manual_fill mean |d|", np.abs(d[sch.manual_fill.to_numpy()]).mean().round(1),
      "auto", np.abs(d[~sch.manual_fill.to_numpy()]).mean().round(1))
# один ТС крупно: как выглядит ряд
s = sch[sch.tr_id == sch.tr_id.iloc[0]].head(60)
print(((s.fact - s.plan)).astype(int).tolist())
print(np.diff(s.plan.to_numpy()).astype(int).tolist())
