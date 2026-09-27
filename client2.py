"""
Клиент: сначала строит контекст из traffic_test.csv + schedule_test.csv,
потом спрашивает tr_id / маршрут / остановку / время
и отправляет отфильтрованные точки на /predict.
"""

import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BASE = "http://5.227.60.94:548"
TIMEOUT_SET_CONTEXT = 180
TIMEOUT_PREDICT = 60

# ↓↓↓ ЗДЕСЬ БЫЛИ ЗАМЕНЫ ↓↓↓
TRAFFIC_CSV = "traffic_test.csv"
SCHEDULE_CSV = "schedule_test.csv"
POINTS_CSV = "points.csv"


# ──────────────────────────── САНИТАЙЗЕР ────────────────────────────

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    if obj is pd.NaT:
        return None
    return obj


# ──────────────────────────── ВВОД ────────────────────────────

def ask_str(prompt):
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("  пусто, попробуй ещё раз")


def ask_time(prompt):
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
        print(f"  не могу разобрать '{raw}'")


# ──────────────────────────── ЭТАП 1: HEALTH ────────────────────────────

def do_health():
    print("=" * 60)
    print("ЭТАП 1: GET /health")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        print(f"HTTP {r.status_code}")
        print(r.json())
    except requests.RequestException as e:
        print(f"❌ {e}")
        sys.exit(1)


# ──────────────────────────── ЭТАП 2: SET_CONTEXT ────────────────────────────

def do_set_context():
    print()
    print("=" * 60)
    print("ЭТАП 2: POST /set_context")
    print("=" * 60)

    print(f"[traffic] читаю {TRAFFIC_CSV}")
    traffic_df = pd.read_csv(TRAFFIC_CSV)
    print(f"[traffic] прочитано {len(traffic_df)} строк, колонки: {list(traffic_df.columns)}")

    print(f"[schedule] читаю {SCHEDULE_CSV}")
    schedule_df = pd.read_csv(SCHEDULE_CSV)
    print(f"[schedule] прочитано {len(schedule_df)} строк, колонки: {list(schedule_df.columns)}")

    # обязательные колонки traffic
    req_t = ["tr_id", "event_time", "lon", "lat"]
    miss_t = [c for c in req_t if c not in traffic_df.columns]
    if miss_t:
        print(f"❌ в {TRAFFIC_CSV} нет колонок: {miss_t}")
        sys.exit(1)

    # обязательные колонки schedule
    req_s = ["tr_id", "tt_action_item_id", "time_begin"]
    miss_s = [c for c in req_s if c not in schedule_df.columns]
    if miss_s:
        print(f"❌ в {SCHEDULE_CSV} нет колонок: {miss_s}")
        sys.exit(1)

    # чистка traffic
    traffic_df["tr_id"] = pd.to_numeric(traffic_df.tr_id, errors="coerce")
    traffic_df["lon"] = pd.to_numeric(traffic_df.lon, errors="coerce")
    traffic_df["lat"] = pd.to_numeric(traffic_df.lat, errors="coerce")
    before = len(traffic_df)
    traffic_df = traffic_df.dropna(subset=["tr_id", "lon", "lat", "event_time"])
    print(f"[traffic] после чистки: {len(traffic_df)} (выкинуто {before - len(traffic_df)})")
    traffic_df["tr_id"] = traffic_df.tr_id.astype(np.int64)

    # чистка schedule
    schedule_df["tr_id"] = pd.to_numeric(schedule_df.tr_id, errors="coerce")
    schedule_df["tt_action_item_id"] = pd.to_numeric(schedule_df.tt_action_item_id, errors="coerce")
    before = len(schedule_df)
    schedule_df = schedule_df.dropna(subset=["tr_id", "tt_action_item_id", "time_begin"])
    print(f"[schedule] после чистки: {len(schedule_df)} (выкинуто {before - len(schedule_df)})")
    schedule_df["tr_id"] = schedule_df.tr_id.astype(np.int64)
    schedule_df["tt_action_item_id"] = schedule_df.tt_action_item_id.astype(np.int64)

    if len(traffic_df) == 0 or len(schedule_df) == 0:
        print("❌ после чистки остались пустые данные")
        sys.exit(1)

    traffic_cols = [c for c in [
        "tr_id", "event_time", "receive_time", "lon", "lat",
        "speed", "heading", "location_valid",
    ] if c in traffic_df.columns]

    schedule_cols = [c for c in [
        "tt_action_item_id", "tr_id", "time_begin",
        "geom", "building_address", "manual_fill",
    ] if c in schedule_df.columns]

    traffic_payload = sanitize(traffic_df[traffic_cols].to_dict(orient="records"))
    schedule_payload = sanitize(schedule_df[schedule_cols].to_dict(orient="records"))

    print(f"[traffic]  к отправке: {len(traffic_payload)}")
    print(f"[schedule] к отправке: {len(schedule_payload)}")
    print(f"[traffic]  образец: {traffic_payload[0]}")
    print(f"[schedule] образец: {schedule_payload[0]}")

    try:
        r = requests.post(
            f"{BASE}/set_context",
            json={"traffic": traffic_payload, "schedule": schedule_payload},
            timeout=TIMEOUT_SET_CONTEXT,
        )
    except requests.exceptions.InvalidJSONError as e:
        print(f"❌ JSON невалиден: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"HTTP {r.status_code}")
    if r.status_code >= 400:
        print("❌ Сервер вернул ошибку:")
        print(r.text[:3000])
        sys.exit(1)

    data = r.json()
    print(f"✅ Контекст построен: n_vehicles={data.get('n_vehicles')}, "
          f"elapsed={data.get('elapsed_sec')} s")


# ──────────────────────────── ЗАГРУЗКА POINTS ────────────────────────────

def load_points(path=POINTS_CSV):
    print()
    print(f"[points] читаю {path}")
    df = pd.read_csv(path)
    print(f"[points] прочитано {len(df)} строк, колонки: {list(df.columns)}")

    required = ["sample_id", "tr_id", "T", "target_stop_id",
                "target_time_begin", "cur_dev_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ нет колонок: {missing}")
        sys.exit(1)

    df["tr_id"] = df["tr_id"].astype(str)
    df["target_stop_id"] = df["target_stop_id"].astype(str)
    df["cur_dev_s"] = pd.to_numeric(df["cur_dev_s"], errors="coerce")
    df["T_dt"] = pd.to_datetime(df["T"], errors="coerce")
    df["target_time_begin_dt"] = pd.to_datetime(df["target_time_begin"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["cur_dev_s", "T_dt", "target_time_begin_dt"])
    print(f"[points] после чистки: {len(df)} (выкинуто {before - len(df)})")
    if len(df) == 0:
        print("❌ после чистки ничего не осталось")
        sys.exit(1)
    return df.reset_index(drop=True)


# ──────────────────────────── ФИЛЬТРАЦИЯ ────────────────────────────

def filter_rows(df, tr_id, route_id, stop_id, when):
    print()
    print("─" * 60)
    print(f"  Фильтр: tr_id={tr_id!r}, route_id={route_id!r}, "
          f"target_stop_id={stop_id!r}, T≈{when}")
    print("─" * 60)

    cur = df.copy()
    print(f"старт: {len(cur)}")

    # 1) tr_id
    exact = cur[cur["tr_id"] == tr_id]
    if len(exact) > 0:
        cur = exact
        print(f"после tr_id == {tr_id!r}: {len(cur)}")
    else:
        prefix = cur["tr_id"].str.split("_").str[0] == tr_id
        if prefix.any():
            cur = cur[prefix]
            print(f"после tr_id startswith {tr_id!r}: {len(cur)}")
        else:
            print(f"⚠️  ни одна строка не совпала с tr_id={tr_id!r}")
            print(f"   примеры: {cur['tr_id'].head(5).tolist()}")
            sys.exit(1)

    # 2) маршрут (если колонка есть)
    route_col = None
    for c in ("route_id", "route_num", "route"):
        if c in cur.columns:
            route_col = c
            break
    if route_col is None:
        print("⚠️  колонки с маршрутом нет — пропускаю")
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
        print("❌ после фильтрации пусто")
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
                  f"откл={delta[idx]})")
        else:
            target_ts = pd.Timestamp(when)
            delta = (cur["T_dt"] - target_ts).abs()
            window = pd.Timedelta(minutes=10)
            narrowed = cur[delta <= window]
            if len(narrowed) == 0:
                idxs = delta.nsmallest(min(5, len(cur))).index
                cur = cur.loc[idxs]
                print(f"±10 мин пусто → беру 5 ближайших")
            else:
                cur = narrowed
                print(f"после окна ±10 мин от {when}: {len(cur)}")

    return cur.reset_index(drop=True)


# ──────────────────────────── ЭТАП 3: PREDICT ────────────────────────────

def do_predict():
    print()
    print("=" * 60)
    print("ЭТАП 3: POST /predict")
    print("=" * 60)

    tr_id = ask_str("Номер автобуса (tr_id): ")
    route_id = ask_str("Номер маршрута: ")
    stop_id = ask_str("Номер остановки (target_stop_id): ")
    when = ask_time("Время (YYYY-MM-DD HH:MM:SS, 'now', или '-'): ")

    df = load_points()
    selected = filter_rows(df, tr_id, route_id, stop_id, when)

    print()
    print(f"✅ Отобрано {len(selected)} строк")

    payload = selected[[
        "sample_id", "tr_id", "T", "target_stop_id",
        "target_time_begin", "cur_dev_s",
    ]].to_dict(orient="records")

    for row in payload:
        row["sample_id"] = str(row["sample_id"])
        row["tr_id"] = str(row["tr_id"])
        row["T"] = str(row["T"])
        row["target_stop_id"] = str(row["target_stop_id"])
        row["target_time_begin"] = str(row["target_time_begin"])
        row["cur_dev_s"] = float(row["cur_dev_s"])

    try:
        r = requests.post(f"{BASE}/predict",
                          json={"points": payload}, timeout=TIMEOUT_PREDICT)
    except requests.RequestException as e:
        print(f"❌ {e}")
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


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 60)
    print("  ПРЕДСКАЗАНИЕ ЗАДЕРЖКИ АВТОБУСА")
    print(f"  traffic={TRAFFIC_CSV}  schedule={SCHEDULE_CSV}  points={POINTS_CSV}")
    print("=" * 60)

    do_health()
    do_set_context()
    do_predict()

    print()
    print("✅ Готово")


if __name__ == "__main__":
    main()