"""Проверка сборки выборок и простых эвристик (MAE) на train-расширении и test."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.data import DATA, ROOT, load_points  # noqa: E402
from ml.dataset import augmented_points, make_context  # noqa: E402
from ml.features import build_table  # noqa: E402

t0 = time.time()
ctx_tr, prof_tr, sch_tr = make_context(DATA / "train/traffic.csv", DATA / "train/schedule.csv")
ctx_te, prof_te, _ = make_context(DATA / "test/traffic.csv", DATA / "validate/schedule_plan.csv")
print("ctx", round(time.time() - t0, 1))

aug = augmented_points(sch_tr, {"train"})
print("aug", len(aug), aug.slot_split.value_counts().to_dict())
lt = load_points(DATA / "labels/labels_train.csv")
lt = lt[lt.tr_id < 9_000_000]
m = lt.merge(aug, on=["tr_id", "T_s"], suffixes=("", "_a"))
print("label match", len(m), "/", len(lt),
      "target eq", round((m.target_delay_s == m.target_delay_s_a).mean(), 3),
      "stop eq", round((m.target_stop_id == m.target_stop_id_a).mean(), 3),
      "curdev eq", round((m.cur_dev_s == m.cur_dev_s_a).mean(), 3))

te = load_points(DATA / "labels/labels_test.csv")
Xa = build_table(aug, ctx_tr, prof_tr)
Xt = build_table(te, ctx_te, prof_te)
print("features", round(time.time() - t0, 1))
ya, yt = aug.target_delay_s.to_numpy(), te.target_delay_s.to_numpy()
for c in ["cur_dev", "dev_now", "dev_interp", "dev_mm_mean_120", "dev_last", "mean_dev15"]:
    line = f"{c:12s}"
    for X, y, n in ((Xa, ya, "aug"), (Xt, yt, "test")):
        v = X[c].to_numpy()
        f = np.where(np.isfinite(v), v, X.cur_dev.to_numpy())
        line += f" | {n} MAE {np.abs(y - f).mean():6.1f}"
    print(line)
print("zero", np.abs(ya).mean().round(1), np.abs(yt).mean().round(1))
print(Xt.describe().T[["mean", "50%", "min", "max"]].round(1).to_string())
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts"
out.mkdir(exist_ok=True, parents=True)
Xa.assign(y=ya, tr=aug.tr_id.values, T=aug.T_s.values, split=aug.slot_split.values).to_pickle(out / "aug.pkl")
Xt.assign(y=yt, tr=te.tr_id.values, T=te.T_s.values).to_pickle(out / "test.pkl")
