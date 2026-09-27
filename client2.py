"""
Клиент: спрашивает у пользователя tr_id, маршрут, остановку и время,
отправляет на /predict только подходящие строки.

Особенности:
  - tr_id и route_id трактуются как СТРОКИ (могут быть составные, напр. "122048_1767732900")
  - target_stop_id — тоже строка (крупные числа)
  - время фильтруется гибко: 'now', точное значение, '-' (пропустить)
"""

import sys
from datetime import datetime

import pandas as pd
import requests

BASE = "http://5.227.60.94:548"
TIMEOUT = 60


# ──────────────────────────── ВВОД ────────────────────────────

def ask_str(prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("  пусто, попробуй ещё раз")


def ask_time(prompt: str):
    """Возвращает datetime, 'now' или None (не фильтровать)."""
    while True:
        raw = input(prompt).strip()
        low = raw.lower()
        if low in ("-", "none", "skip", ""):
            print("  время не фильтруем")
            return None
        if low == "now":
            print(f"  использую текущее время: {datetime.now():%Y-%m-%d %H:%M:%S}")
            return "now"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        print(f"  не могу разобрать '{raw}'. Форматы: "
              f"'2026-09-27 14:30:00', 'now', '-' (пропустить)")


# ──────────────────────────── ЗАГРУЗКА ────────────────────────────

def load_points(path="points.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[points] прочитано {len(df)} строк, колонки: {list(df.columns)}")

    required = ["sample_id", "tr_id", "T", "target_stop_id",
                "target_time_begin", "cur_dev_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ нет колонок: {missing}")
        sys.exit(1)

    # tr_id / target_stop_id — как строки
    df["tr_id"] = df["tr_id"].astype(str)
    df["target_stop_id"] = df["target_stop_id"].astype(str)

    # cur_dev_s — числовой
    df["cur_dev_s"] = pd.to_numeric(df["cur_dev_s"], errors="coerce")

    # ВАЖНО: обращаемся через df["T"], а не df.T (это transpose!)
    df["T_dt"] = pd.to_datetime(df["T"], errors="coerce")
    df["target_time_begin_dt"] = pd.to_datetime(df["target_time_begin"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["cur_dev_s", "T_dt", "target_time_begin_dt"])
    print(f"[points] после чистки NaN: {len(df)} (выкинуто {before - len(df)})")

    if len(df) == 0:
        print("❌ после чистки ничего не осталось")
        sys.exit(1)

    return df.reset_index(drop=True)


# ──────────────────────────── ФИЛЬТРАЦИЯ ────────────────────────────

def filter_rows(df: pd.DataFrame, tr_id: str, route_id: str,
                stop_id: str, when) -> pd.DataFrame:
    print()
    print("─" * 60)
    print(f"  Фильтр: tr_id={tr_id!r}, route_id={route_id!r}, "
          f"target_stop_id={stop_id!r}, T≈{when}")
    print("─" * 60)

    cur = df.copy()
    print(f"старт: {len(cur)}")

    # 1) автобус — точное совпадение по строке, либо префикс до '_'
    tr_exact = cur[cur["tr_id"] == tr_id]
    if len(tr_exact) > 0:
        cur = tr_exact
        print(f"после tr_id == {tr_id!r}: {len(cur)}")
    else:
        # вдруг передали только числовую часть, а в файле "122048_..."
        prefix_mask = cur["tr_id"].str.split("_").str[0] == tr_id
        if prefix_mask.any():
            cur = cur[prefix_mask]
            print(f"после tr_id startswith {tr_id!r}: {len(cur)}")
        else:
            print(f"⚠️  ни одна строка не совпала с tr_id={tr_id!r}")
            print(f"   примеры tr_id в файле: {cur['tr_id'].head(5).tolist()}")
            sys.exit(1)

    # 2) маршрут — если колонка есть
    route_col = None
    for c in ("route_id", "route_num", "route"):
        if c in cur.columns:
            route_col = c
            break
    if route_col is None:
        print(f"⚠️  колонки с маршрутом нет. "
              f"Если маршрут зашит в tr_id — фильтр уже отработал.")
    else:
        cur[route_col] = cur[route_col].astype(str)
        before = len(cur)
        cur = cur[cur[route_col] == route_id]
        print(f"после {route_col} == {route_id!r}: {len(cur)} (было {before})")

    # 3) остановка
    before = len(cur)
    cur = cur[cur["target_stop_id"] == stop_id]
    print(f"после target_stop_id == {stop_id!r}: {len(cur)} (было {before})")

    if len(cur) == 0:
        print("❌ после фильтрации не осталось строк")
        print(f"   tr_id в файле: {df['tr_id'].unique()[:10].tolist()}")
        print(f"   target_stop_id в файле: {df['target_stop_id'].unique()[:10].tolist()}")
        sys.exit(1)

    # 4) время
    if when is not None:
        if when == "now":
            now = pd.Timestamp.now()
            delta = (cur["T_dt"] - now).abs()
            idx = delta.idxmin()
            cur = cur.loc[[idx]]
            print(f"после 'now': выбрана ближайшая (T={cur.iloc[0]['T']}, "
                  f"отклонение={delta[idx]})")
        else:
            target_ts = pd.Timestamp(when)
            delta = (cur["T_dt"] - target_ts).abs()
            window = pd.Timedelta(minutes=10)
            narrowed = cur[delta <= window]
            if len(narrowed) == 0:
                idxs = delta.nsmallest(min(5, len(cur))).index
                cur = cur.loc[idxs]
                print(f"±10 мин пусто → беру 5 ближайших по T")
            else:
                cur = narrowed
                print(f"после окна ±10 мин от {when}: {len(cur)}")

    return cur.reset_index(drop=True)


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 60)
    print("  ПРЕДСКАЗАНИЕ ЗАДЕРЖКИ АВТОБУСА")
    print("=" * 60)

    tr_id = ask_str("Номер автобуса (tr_id): ")
    route_id = ask_str("Номер маршрута: ")
    stop_id = ask_str("Номер остановки (target_stop_id): ")
    when = ask_time("Время (YYYY-MM-DD HH:MM:SS, 'now', или '-'): ")

    df = load_points()
    selected = filter_rows(df, tr_id, route_id, stop_id, when)

    print()
    print(f"✅ Отобрано {len(selected)} строк")
    first = selected.iloc[0]
    print(f"   образец: sample_id={first['sample_id']} "
          f"tr_id={first['tr_id']} T={first['T']} "
          f"target_stop_id={first['target_stop_id']} "
          f"cur_dev_s={first['cur_dev_s']}")

    payload = selected[[
        "sample_id", "tr_id", "T", "target_stop_id",
        "target_time_begin", "cur_dev_s",
    ]].to_dict(orient="records")

    # явное приведение типов, чтобы никакие numpy.* не просочились
    for row in payload:
        row["sample_id"] = str(row["sample_id"])
        row["tr_id"] = str(row["tr_id"])
        row["T"] = str(row["T"])
        row["target_stop_id"] = str(row["target_stop_id"])
        row["target_time_begin"] = str(row["target_time_begin"])
        row["cur_dev_s"] = float(row["cur_dev_s"])

    try:
        r = requests.post(f"{BASE}/predict",
                          json={"points": payload}, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
        sys.exit(1)

    print(f"HTTP {r.status_code}")
    if r.status_code >= 400:
        print("❌ Сервер вернул ошибку:")
        print(r.text[:2000])
        sys.exit(1)

    data = r.json()
    print()
    print("=" * 60)
    print(f"  ПРОГНОЗ: tr_id={tr_id}  маршрут={route_id}  остановка={stop_id}")
    print("=" * 60)
    for sid, p in zip(data["sample_id"], data["prediction"]):
        print(f"   {sid}: {p} s")


if __name__ == "__main__":
    main()