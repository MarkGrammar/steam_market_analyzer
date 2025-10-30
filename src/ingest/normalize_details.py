# src/ingest/normalize_details.py
import json
from pathlib import Path
import pandas as pd
from src.utils.logging import get_logger

log = get_logger("normalize_details")

IN_DIR = Path("data/raw/details")
OUT_DIR = Path("data/processed")
OUT_APPS = OUT_DIR / "apps.csv"
OUT_TAGS = OUT_DIR / "tags.csv"
OUT_GENRES = OUT_DIR / "genres.csv"

def load_combo(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apps_rows, tags_rows, genres_rows = [], [], []

    files = sorted(IN_DIR.glob("*.json"))
    for p in files:
        combo = load_combo(p)
        app_id = combo.get("app_id")
        store = combo.get("store", {})
        spy = combo.get("steamspy", {})

        # --- store alanları
        s_ok = store.get("success") and store.get("data")
        name = None
        release_date = None
        price = None
        is_free = None
        positive = None
        negative = None
        genres = []
        if s_ok:
            d = store["data"]
            name = d.get("name")
            is_free = d.get("is_free", False)
            # reviewlar bazen store’da aggregate gelmez, SteamSpy daha tutarlı
            if isinstance(d.get("genres"), list):
                genres = [g.get("description") for g in d["genres"] if g.get("description")]
            # fiyat
            pc = d.get("price_overview") or {}
            if pc:
                price = (pc.get("final", 0) or 0) / 100.0

            # tarih
            rd = d.get("release_date") or {}
            release_date = rd.get("date")

        # --- steamspy alanları
        positive = spy.get("positive") or positive
        negative = spy.get("negative") or negative
        owners_low = None
        owners_high = None
        owners_str = spy.get("owners")
        if owners_str and isinstance(owners_str, str) and "-" in owners_str:
            lo, hi = owners_str.split("-")
            try:
                owners_low = int(lo.replace(",", ""))
                owners_high = int(hi.replace(",", ""))
            except:
                pass

        # tags (steamspy)
        tags = []
        if isinstance(spy.get("tags"), dict):
            tags = list(spy["tags"].keys())

        # app satırı
        apps_rows.append({
            "app_id": app_id,
            "name": name,
            "release_date": release_date,
            "is_free": int(bool(is_free)),
            "price": price,
            "positive": positive,
            "negative": negative,
            "owners_low": owners_low,
            "owners_high": owners_high,
        })

        # genres satırları
        for g in genres:
            genres_rows.append({"app_id": app_id, "genre": g})

        # tags satırları
        for t in tags:
            tags_rows.append({"app_id": app_id, "tag": t})

    pd.DataFrame(apps_rows).to_csv(OUT_APPS, index=False)
    pd.DataFrame(tags_rows).to_csv(OUT_TAGS, index=False)
    pd.DataFrame(genres_rows).to_csv(OUT_GENRES, index=False)

    log.info(f"apps:   {OUT_APPS}  ({len(apps_rows)})")
    log.info(f"tags:   {OUT_TAGS}  ({len(tags_rows)})")
    log.info(f"genres: {OUT_GENRES} ({len(genres_rows)})")

if __name__ == "__main__":
    main()
