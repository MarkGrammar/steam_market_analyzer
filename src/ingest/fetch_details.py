# src/ingest/fetch_details.py
import time, json
import requests
import pandas as pd
from pathlib import Path
from src.utils.io import write_json
from src.utils.logging import get_logger

log = get_logger("fetch_details")

STORE_API = "https://store.steampowered.com/api/appdetails"
STEAMSPY = "https://steamspy.com/api.php"
APP_LIST = Path("data/raw/app_list.csv")
OUT_DIR = Path("data/raw/details")

def fetch_store(app_id: int):
    r = requests.get(STORE_API, params={"appids": app_id, "cc": "us", "l": "en"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get(str(app_id), {})

def fetch_spy(app_id: int):
    r = requests.get(STEAMSPY, params={"request": "appdetails", "appid": app_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def main(limit: int | None = None, sleep_sec: float = 0.8):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(APP_LIST)
    if limit:
        df = df.head(limit)
    for i, row in df.iterrows():
        app_id = int(row["app_id"])
        out_path = OUT_DIR / f"{app_id}.json"
        if out_path.exists():
            continue
        try:
            store = fetch_store(app_id)
            spy = fetch_spy(app_id)
            combo = {"app_id": app_id, "store": store, "steamspy": spy}
            write_json(combo, str(out_path))
            if i % 20 == 0:
                log.info(f"{i}/{len(df)} saved: {app_id}")
            time.sleep(sleep_sec)
        except Exception as e:
            log.warning(f"failed {app_id}: {e}")
            time.sleep(1.5)

if __name__ == "__main__":
    main()
