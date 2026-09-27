"""Клиент для проверки API."""
import requests
import pandas as pd

BASE = "http://localhost:8000"

# 1. health
print("health:", requests.get(f"{BASE}/health").json())

# 2. set_context — из CSV
traffic_df  = pd.read_csv("traffic_valid.csv")
schedule_df = pd.read_csv("schedule_plan_valid.csv")

traffic_df["tr_id"]  = pd.to_numeric(traffic_df.tr_id,  errors="coerce")
schedule_df["tr_id"] = pd.to_numeric(schedule_df.tr_id, errors="coerce")
traffic_df  = traffic_df.dropna(subset=["tr_id"])
schedule_df = schedule_df.dropna(subset=["tr_id"])

print(f"traffic: {traffic_df.shape}, schedule: {schedule_df.shape}")

traffic_payload = traffic_df[[
    "tr_id", "event_time", "receive_time", "lon", "lat",
    "speed", "heading", "location_valid",
]].to_dict(orient="records")

schedule_payload = schedule_df[[
    "tt_action_item_id", "tr_id", "time_begin",
    "geom", "building_address", "manual_fill",
]].to_dict(orient="records")

resp = requests.post(f"{BASE}/set_context", json={
    "traffic": traffic_payload,
    "schedule": schedule_payload,
}, timeout=120)
print("set_context:", resp.status_code, resp.json())

# 3. predict — из points.csv
points_df = pd.read_csv("points.csv")
points_df["tr_id"] = pd.to_numeric(points_df.tr_id, errors="coerce")
points_df = points_df.dropna(subset=["tr_id"])

points_payload = points_df[[
    "sample_id", "tr_id", "T", "target_stop_id",
    "target_time_begin", "cur_dev_s",
]].to_dict(orient="records")

resp = requests.post(f"{BASE}/predict", json={"points": points_payload}, timeout=60)
print("predict:", resp.status_code)
res = resp.json()
print(f"Получено {len(res['prediction'])} прогнозов")
for sid, p in list(zip(res["sample_id"], res["prediction"]))[:5]:
    print(f"  {sid}: {p} s")
