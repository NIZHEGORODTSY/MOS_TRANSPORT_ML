"""Проверка map matching (train): задержка из привязки в момент факта vs fact − plan."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.arrivals import map_match  # noqa: E402
from ml.data import DATA, load_schedule, load_traffic  # noqa: E402

t0 = time.time()
tf = load_traffic(DATA / "train/traffic.csv")
sch = load_schedule(DATA / "train/schedule.csv")
for kw in (dict(), dict(w_time=0.05), dict(w_time=0.3), dict(w_head=0.0)):
    m = map_match(sch, tf, **kw)
    errs, found = [], 0
    for tr, s in sch.groupby("tr_id"):
        p = m[m.tr_id == tr]
        t = p.t.to_numpy()
        for f, pl in zip(s.fact.to_numpy(), s.plan.to_numpy()):
            i = np.searchsorted(t, f)
            if i < len(t) and t[i] - f < 30 and np.isfinite(p.mm_dev.iat[i]) and p.mm_dist.iat[i] < 60:
                errs.append(p.mm_dev.iat[i] - (f - pl))
                found += 1
    e = np.array(errs)
    print(kw, f"cover {found / len(sch):.2f} med {np.median(e):.1f} MAE {np.abs(e).mean():.1f} "
          f"p50 {np.median(np.abs(e)):.1f} p90 {np.quantile(np.abs(e), .9):.1f} "
          f"dist<60 frac {(m.mm_dist < 60).mean():.2f}  {time.time() - t0:.1f}s")
