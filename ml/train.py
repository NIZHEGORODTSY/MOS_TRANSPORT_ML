"""Обучение и оценка модели задержек; формирование submission.csv.

Примеры:
    python -m ml.train --eval                 # CV на train + проверка на отложенном test
    python -m ml.train --final                # обучение на train+test, прогноз validate
    python -m ml.train --final --use-manual-fill

Протокол без утечек:
  * признаки строятся только из телеметрии (event_time ≤ T), планового расписания
    (validate/schedule_plan.csv — без фактов) и подсказки cur_dev_s;
  * факты расписания используются только как метки (и cur_dev_s для расширенных точек,
    так же как его считают организаторы);
  * при оценке на test модель не видит ни одной остановки из окон test/validate.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import DATA, ROOT, load_points, load_schedule
from .dataset import augmented_points, make_context
from .features import FEATURES, OPTIONAL_FEATURES, build_sequences, build_table
from .models import CB_PARAMS, ModelBundle, NetConfig, Preprocessor, predict_net, train_catboost, train_net

ART = ROOT / "artifacts"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def pseudo_score(y, p, cur) -> float:
    """Оценка как на платформе; MAE_TARGET подобран так, чтобы baseline cur_dev давал 0.40."""
    mz, mc = mae(y, 0), mae(y, cur)
    target = mz - (mz - mc) / 0.40
    return float(np.clip((mz - mae(y, p)) / (mz - target), 0, 1))


class Pipeline:
    """Общий контекст + построение признаков для любых прогнозных точек."""

    def __init__(self):
        log("контекст: телеметрия + плановое расписание (без фактов)")
        # телеметрия реальных ТС в train/test/validate идентична; расписание — только план
        self.ctx, self.prof, _ = make_context(DATA / "test/traffic.csv", DATA / "validate/schedule_plan.csv")
        self.sch_fact = load_schedule(DATA / "train/schedule.csv")  # только для меток

    def features(self, pts: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return build_table(pts, self.ctx, self.prof), build_sequences(pts, self.ctx)


def fit_models(X: pd.DataFrame, S: np.ndarray, y: np.ndarray, feats: list[str], cfg: NetConfig):
    prep = Preprocessor().fit(X[feats])
    tab, seq = prep.tab(X[feats]), prep.seq(S)
    nets = [train_net(tab, seq, y, cfg, s) for s in cfg.seeds]
    cb = train_catboost(X[feats], y)
    return prep, nets, cb


def predict_models(prep, nets, cb, X, S, feats):
    return predict_net(nets, prep.tab(X[feats]), prep.seq(S)), cb.predict(X[feats])


def run_eval(pl: Pipeline, feats: list[str], cfg: NetConfig) -> float:
    """CV на расширенном train + отложенный test. Возвращает лучший вес смеси по OOF."""
    tr = augmented_points(pl.sch_fact, {"train"})
    te = load_points(DATA / "labels/labels_test.csv")
    Xtr, Str = pl.features(tr)
    Xte, Ste = pl.features(te)
    ytr, yte = tr.target_delay_s.to_numpy(), te.target_delay_s.to_numpy()
    log(f"train (расширенный) {len(tr)} точек, test {len(te)} точек, признаков {len(feats)}")

    groups = tr.tr_id.to_numpy() * 100 + (tr.T_s.to_numpy() // 3600).astype(int) % 100
    oof_net, oof_cb = np.zeros(len(tr)), np.zeros(len(tr))
    for f, (a, b) in enumerate(GroupKFold(5).split(Xtr, ytr, groups)):
        prep, nets, cb = fit_models(Xtr.iloc[a], Str[a], ytr[a], feats, cfg)
        oof_net[b], oof_cb[b] = predict_models(prep, nets, cb, Xtr.iloc[b], Str[b], feats)
        log(f"  fold {f}: net {mae(ytr[b], oof_net[b]):.1f}  cb {mae(ytr[b], oof_cb[b]):.1f}")
    ws = np.linspace(0, 1, 11)
    cv = [mae(ytr, w * oof_net + (1 - w) * oof_cb) for w in ws]
    w = float(ws[int(np.argmin(cv))])

    prep, nets, cb = fit_models(Xtr, Str, ytr, feats, cfg)
    p_net, p_cb = predict_models(prep, nets, cb, Xte, Ste, feats)
    p = w * p_net + (1 - w) * p_cb
    cur = te.cur_dev_s.to_numpy()
    cur_tr = tr.cur_dev_s.to_numpy()
    rows = [
        ("zero", mae(ytr, 0), mae(yte, 0)),
        ("cur_dev (baseline)", mae(ytr, cur_tr), mae(yte, cur)),
        ("0.55·cur_dev", mae(ytr, 0.55 * cur_tr), mae(yte, 0.55 * cur)),
        ("PyTorch GRU+MLP", mae(ytr, oof_net), mae(yte, p_net)),
        ("CatBoost", mae(ytr, oof_cb), mae(yte, p_cb)),
        (f"ансамбль (w_net={w:.1f})", min(cv), mae(yte, p)),
    ]
    print("\n{:28s} {:>10s} {:>10s}".format("модель", "CV MAE", "test MAE"))
    for name, a, b in rows:
        print(f"{name:28s} {a:10.1f} {b:10.1f}")
    print(f"* test score (формула платформы, MAE_TARGET оценён по baseline=0.40): "
          f"cur_dev {pseudo_score(yte, cur, cur):.2f}, net {pseudo_score(yte, p_net, cur):.2f}, "
          f"cb {pseudo_score(yte, p_cb, cur):.2f}, ансамбль {pseudo_score(yte, p, cur):.2f}")
    imp = pd.Series(cb.get_feature_importance(), index=feats).sort_values(ascending=False)
    print("\nCatBoost: топ-15 признаков\n" + imp.head(15).round(1).to_string())
    return w


def run_final(pl: Pipeline, feats: list[str], cfg: NetConfig, w: float, tag: str) -> None:
    tr = augmented_points(pl.sch_fact, {"train", "test"})
    va = load_points(DATA / "validate/points.csv")
    Xtr, Str = pl.features(tr)
    Xva, Sva = pl.features(va)
    ytr = tr.target_delay_s.to_numpy()
    log(f"финальное обучение: {len(tr)} точек (train+test), w_net={w}")
    prep, nets, cb = fit_models(Xtr, Str, ytr, feats, cfg)
    bundle = ModelBundle(feats, prep, nets, cb, w, cfg)
    out = ART / f"model{tag}"
    bundle.save(out)
    pred = bundle.predict(Xva, Sva)
    sub = pd.read_csv(DATA / "sample_submission.csv", sep=";")
    m = dict(zip(va.sample_id, pred))
    sub["prediction"] = sub.sample_id.map(m).round(1)
    assert sub.prediction.notna().all() and sub.sample_id.is_unique and len(sub) == len(va)
    path = ART / f"submission{tag}.csv"
    sub.to_csv(path, sep=";", index=False)
    log(f"модель → {out}\nsubmission → {path}  (mean {sub.prediction.mean():.1f}, "
        f"corr с cur_dev {np.corrcoef(sub.prediction, va.set_index('sample_id').loc[sub.sample_id].cur_dev_s)[0, 1]:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--use-manual-fill", action="store_true")
    ap.add_argument("--w-net", type=float, default=None, help="вес нейросети в смеси (иначе — по CV)")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    feats = [f for f in FEATURES if a.use_manual_fill or f not in OPTIONAL_FEATURES]
    cfg = NetConfig(epochs=a.epochs)
    tag = "_mf" if a.use_manual_fill else ""
    pl = Pipeline()
    w = a.w_net
    if a.eval or w is None:
        w = run_eval(pl, feats, cfg)
    if a.final:
        run_final(pl, feats, cfg, w, tag)


if __name__ == "__main__":
    main()
