# src/ingest/fetch_app_list.py
import requests
import pandas as pd
from pathlib import Path

STEAMSPY = "https://steamspy.com/api.php"
OUT_CSV = Path("data/raw/app_list.csv")

def main(limit: int = 2000):
    print(f"[info] Fetching up to {limit} apps from SteamSpy...")

    # SteamSpy "all" → tüm oyunların kısa özeti (~50k oyun)
    resp = requests.get(STEAMSPY, params={"request": "all"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for _, v in data.items():
        rows.append({"app_id": int(v["appid"]), "name": v.get("name", "")})

    df = pd.DataFrame(rows).drop_duplicates("app_id").sort_values("app_id")

    # isteğe göre limitle
    if limit and limit < len(df):
        df = df.sample(limit, random_state=42).sort_values("app_id")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[done] Saved {len(df)} apps → {OUT_CSV}")

if __name__ == "__main__":
    main()
