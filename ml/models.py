"""Модели: PyTorch-сеть (GRU по истории телеметрии + MLP по табличным признакам) и CatBoost.

Обе модели учатся на MAE (L1) — это метрика соревнования. Нейросеть — ансамбль из
нескольких сидов; итоговый прогноз — взвешенная смесь нейросети и CatBoost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import SEQ_CHANNELS

Y_SCALE = 100.0  # таргет в сотнях секунд — удобный масштаб для сети

# фиксированные масштабы каналов последовательности
SEQ_SCALE = {
    "dev_now": 300.0, "dev_interp": 300.0, "cur_speed": 30.0, "spd_mean_120": 30.0,
    "stop_frac_120": 1.0, "d_next": 500.0, "age_pos": 300.0, "dist_120": 1000.0,
}


# ------------------------------------------------------------------ preprocessing
@dataclass
class Preprocessor:
    """Робастная нормализация табличных признаков + маски пропусков; масштабирование последовательностей."""

    columns: list[str] = field(default_factory=list)
    center: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    nan_cols: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns = list(X.columns)
        for c in self.columns:
            v = X[c].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                self.center[c], self.scale[c] = 0.0, 1.0
                continue
            q1, q5, q9 = np.quantile(v, [0.1, 0.5, 0.9])
            self.center[c] = float(q5)
            self.scale[c] = float(max(q9 - q1, 1e-3))
        self.nan_cols = [c for c in self.columns if X[c].isna().any()]
        return self

    def tab(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.columns:
            v = (X[c].to_numpy(np.float64) - self.center[c]) / self.scale[c]
            cols.append(np.clip(np.nan_to_num(v, nan=0.0), -5, 5))
        for c in self.nan_cols:
            cols.append(X[c].isna().to_numpy(np.float64))
        return np.stack(cols, axis=1).astype(np.float32)

    @staticmethod
    def seq(S: np.ndarray) -> np.ndarray:
        """[N, L, C] → [N, L, 2C]: масштабированные значения + маска пропуска."""
        sc = np.array([SEQ_SCALE[c] for c in SEQ_CHANNELS], dtype=np.float32)
        miss = ~np.isfinite(S)
        v = np.clip(np.nan_to_num(S / sc, nan=0.0), -5, 5)
        return np.concatenate([v, miss.astype(np.float32)], axis=-1).astype(np.float32)

    def to_json(self) -> dict:
        return dict(columns=self.columns, center=self.center, scale=self.scale, nan_cols=self.nan_cols)

    @classmethod
    def from_json(cls, d: dict) -> "Preprocessor":
        return cls(**d)


# ------------------------------------------------------------------ network
class DelayNet(nn.Module):
    """GRU-энкодер истории состояния ТС + MLP по табличным признакам → задержка (сотни секунд)."""

    def __init__(self, n_tab: int, n_seq: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.seq_in = nn.Linear(n_seq, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.tab = nn.Sequential(
            nn.Linear(n_tab, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, tab: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(torch.relu(self.seq_in(seq)))
        z = torch.cat([h[-1], self.tab(tab)], dim=1)
        return self.head(z).squeeze(1)


@dataclass
class NetConfig:
    hidden: int = 64
    dropout: float = 0.1
    epochs: int = 40
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seeds: tuple = (0, 1, 2, 3, 4)


def train_net(tab: np.ndarray, seq: np.ndarray, y: np.ndarray, cfg: NetConfig, seed: int) -> DelayNet:
    """Обучение одной сети (L1-loss, AdamW, косинусный LR)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = DelayNet(tab.shape[1], seq.shape[2], cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(y) / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    T, S, Y = torch.from_numpy(tab), torch.from_numpy(seq), torch.from_numpy((y / Y_SCALE).astype(np.float32))
    loss_fn = nn.L1Loss()
    net.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(Y))
        for b in range(0, len(Y), cfg.batch):
            ix = perm[b : b + cfg.batch]
            opt.zero_grad()
            loss = loss_fn(net(T[ix], S[ix]), Y[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    net.eval()
    return net


@torch.no_grad()
def predict_net(nets: list[DelayNet], tab: np.ndarray, seq: np.ndarray) -> np.ndarray:
    T, S = torch.from_numpy(tab), torch.from_numpy(seq)
    return np.mean([n(T, S).numpy() for n in nets], axis=0) * Y_SCALE


# ------------------------------------------------------------------ catboost
CB_PARAMS = dict(loss_function="MAE", depth=6, learning_rate=0.03, iterations=1500,
                 l2_leaf_reg=5.0, random_seed=0, verbose=False, thread_count=-1)


def train_catboost(X: pd.DataFrame, y: np.ndarray, params: dict | None = None):
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(**(params or CB_PARAMS))
    m.fit(X, y)
    return m


# ------------------------------------------------------------------ bundle (save/load)
@dataclass
class ModelBundle:
    """Всё, что нужно для инференса: признаки, препроцессор, сети, CatBoost, вес смеси."""

    features: list[str]
    prep: Preprocessor
    nets: list[DelayNet]
    cb: object
    w_net: float
    net_cfg: NetConfig

    def predict(self, X: pd.DataFrame, S: np.ndarray) -> np.ndarray:
        X = X[self.features]
        p_net = predict_net(self.nets, self.prep.tab(X), self.prep.seq(S))
        p_cb = self.cb.predict(X)
        return self.w_net * p_net + (1 - self.w_net) * p_cb

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = dict(features=self.features, prep=self.prep.to_json(), w_net=self.w_net,
                    net_cfg=self.net_cfg.__dict__, n_tab=len(self.prep.columns) + len(self.prep.nan_cols),
                    n_seq=2 * len(SEQ_CHANNELS), seq_channels=SEQ_CHANNELS)
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        for i, n in enumerate(self.nets):
            torch.save(n.state_dict(), path / f"net_{i}.pt")
        self.cb.save_model(str(path / "catboost.cbm"))

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        from catboost import CatBoostRegressor

        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        cfg = NetConfig(**{k: tuple(v) if k == "seeds" else v for k, v in meta["net_cfg"].items()})
        nets = []
        for f in sorted(path.glob("net_*.pt")):
            n = DelayNet(meta["n_tab"], meta["n_seq"], cfg.hidden, cfg.dropout)
            n.load_state_dict(torch.load(f, map_location="cpu"))
            n.eval()
            nets.append(n)
        cb = CatBoostRegressor()
        cb.load_model(str(path / "catboost.cbm"))
        return cls(meta["features"], Preprocessor.from_json(meta["prep"]), nets, cb, meta["w_net"], cfg)
