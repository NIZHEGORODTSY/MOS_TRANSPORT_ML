"""
Клиент для проверки API предсказания задержек автобусов.

ВАЖНО: серверная схема TrafficRow требует обязательные lon и lat (float).
Поэтому строки без координат отбрасываются до отправки — иначе сервер
возвращает 422.

Запуск:
    python client.py
"""

import math
import sys

import numpy as np
import pandas as pd
import requests

BASE = "http://5.227.60.94:548"
TIMEOUT_SET_CONTEXT = 120
TIMEOUT_PREDICT = 60


# ──────────────────────────── САНИТАЙЗЕР ────────────────────────────

def sanitize(obj):
    """Рекурсивно: NaN/inf → None, np.* → питоновские типы, NaT → None."""
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


# ──────────────────────────── ЗАГРУЗКА CSV ────────────────────────────

def load_traffic(path="traffic_valid.csv"):
    """
    Загружает traffic. Отбрасывает строки, где не парсится tr_id, lon или lat —
    потому что сервер их не примет (схема требует float, не Optional).
    """
    df = pd.read_csv(path)
    print(f"[traffic] прочитано {len(df)} строк, колонки: {list(df.columns)}")

    required = ["tr_id", "event_time", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ в {path} нет обязательных колонок: {missing}")
        sys.exit(1)

    df["tr_id"] = pd.to_numeric(df.tr_id, errors="coerce")
    df["lon"] = pd.to_numeric(df.lon, errors="coerce")
    df["lat"] = pd.to_numeric(df.lat, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["tr_id", "lon", "lat", "event_time"])
    after = len(df)
    print(f"[traffic] после чистки: {after} строк (выкинуто {before - after})")

    if after == 0:
        print("❌ после чистки не осталось ни одной строки traffic")
        sys.exit(1)

    df["tr_id"] = df.tr_id.astype(np.int64)
    return df


def load_schedule(path="schedule_plan_valid.csv"):
    """
    Загружает schedule. Обязательны tr_id, tt_action_item_id, time_begin.
    lon/lat в схеме Optional — их можно оставить None.
    """
    df = pd.read_csv(path)
    print(f"[schedule] прочитано {len(df)} строк, колонки: {list(df.columns)}")

    required = ["tr_id", "tt_action_item_id", "time_begin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ в {path} нет обязательных колонок: {missing}")
        sys.exit(1)

    df["tr_id"] = pd.to_numeric(df.tr_id, errors="coerce")
    df["tt_action_item_id"] = pd.to_numeric(df.tt_action_item_id, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["tr_id", "tt_action_item_id", "time_begin"])
    after = len(df)
    print(f"[schedule] после чистки: {after} строк (выкинуто {before - after})")

    if after == 0:
        print("❌ после чистки не осталось ни одной строки schedule")
        sys.exit(1)

    df["tr_id"] = df.tr_id.astype(np.int64)
    df["tt_action_item_id"] = df.tt_action_item_id.astype(np.int64)
    return df


def load_points(path="points.csv"):
    df = pd.read_csv(path)
    print(f"[points] прочитано {len(df)} строк, колонки: {list(df.columns)}")

    required = ["sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ в {path} нет обязательных колонок: {missing}")
        sys.exit(1)

    df["tr_id"] = pd.to_numeric(df.tr_id, errors="coerce")
    df["cur_dev_s"] = pd.to_numeric(df.cur_dev_s, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["tr_id", "T", "target_time_begin", "cur_dev_s"])
    after = len(df)
    print(f"[points] после чистки: {after} строк (выкинуто {before - after})")

    if after == 0:
        print("❌ после чистки не осталось ни одной строки points")
        sys.exit(1)

    df["tr_id"] = df.tr_id.astype(np.int64)
    return df


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
        print(f"❌ Не удалось подключиться: {e}")
        sys.exit(1)


# ──────────────────────────── ЭТАП 2: SET_CONTEXT ────────────────────────────

def do_set_context():
    print()
    print("=" * 60)
    print("ЭТАП 2: POST /set_context")
    print("=" * 60)

    traffic_df = load_traffic()
    schedule_df = load_schedule()

    traffic_cols = [
        "tr_id", "event_time", "receive_time", "lon", "lat",
        "speed", "heading", "location_valid",
    ]
    schedule_cols = [
        "tt_action_item_id", "tr_id", "time_begin",
        "geom", "building_address", "manual_fill",
    ]

    # Оставляем только те колонки, что реально есть в CSV.
    traffic_cols = [c for c in traffic_cols if c in traffic_df.columns]
    schedule_cols = [c for c in schedule_cols if c in schedule_df.columns]

    traffic_payload = sanitize(traffic_df[traffic_cols].to_dict(orient="records"))
    schedule_payload = sanitize(schedule_df[schedule_cols].to_dict(orient="records"))

    print(f"[traffic]  к отправке: {len(traffic_payload)} записей")
    print(f"[schedule] к отправке: {len(schedule_payload)} записей")

    # образец первой записи — чтобы видеть, что реально уходит
    print("[traffic] образец записи:", traffic_payload[0] if traffic_payload else None)
    print("[schedule] образец записи:", schedule_payload[0] if schedule_payload else None)

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
        print(f"❌ Ошибка запроса: {e}")
        sys.exit(1)

    print(f"HTTP {r.status_code}")
    if r.status_code >= 400:
        print("❌ Сервер вернул ошибку:")
        print(r.text[:3000])
        sys.exit(1)

    data = r.json()
    print(f"✅ Контекст построен: n_vehicles={data.get('n_vehicles')}, "
          f"elapsed={data.get('elapsed_sec')} s")


# ──────────────────────────── ЭТАП 3: PREDICT ────────────────────────────

def do_predict():
    print()
    print("=" * 60)
    print("ЭТАП 3: POST /predict")
    print("=" * 60)

    points_df = load_points()

    points_cols = [
        "sample_id", "tr_id", "T", "target_stop_id",
        "target_time_begin", "cur_dev_s",
    ]
    points_cols = [c for c in points_cols if c in points_df.columns]

    points_payload = sanitize(points_df[points_cols].to_dict(orient="records"))
    print(f"[points] к отправке: {len(points_payload)} записей")
    print("[points] образец записи:", points_payload[0] if points_payload else None)

    try:
        r = requests.post(
            f"{BASE}/predict",
            json={"points": points_payload},
            timeout=TIMEOUT_PREDICT,
        )
    except requests.exceptions.InvalidJSONError as e:
        print(f"❌ JSON невалиден: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
        sys.exit(1)

    print(f"HTTP {r.status_code}")
    if r.status_code >= 400:
        print("❌ Сервер вернул ошибку:")
        print(r.text[:3000])
        sys.exit(1)

    data = r.json()
    sample_ids = data.get("sample_id", [])
    predictions = data.get("prediction", [])
    print(f"✅ Получено {len(predictions)} прогнозов")
    for sid, p in list(zip(sample_ids, predictions))[:10]:
        print(f"   {sid}: {p} s")


# ──────────────────────────── MAIN ────────────────────────────

def main():
    do_health()
    do_set_context()
    do_predict()
    print()
    print("=" * 60)
    print("✅ Готово")
    print("=" * 60)


if __name__ == "__main__":
    main()