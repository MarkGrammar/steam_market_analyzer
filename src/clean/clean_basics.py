# src/clean/clean_basics.py
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
from src.config import MAX_PRICE_EUR, MIN_YEAR
from src.utils.logging import get_logger

log = get_logger("clean_basics")

DB = Path("data/processed/steam.duckdb")
OUT_PARQUET = Path("data/processed/apps_clean.parquet")

def parse_year(s):
    # Steam tarih formatları değişken olabilir, en mantıklısı yıl çekmek
    if pd.isna(s):
        return np.nan
    try:
        # çoğu formatta yıl sonda
        for token in str(s).split():
            if token.isdigit() and len(token) == 4:
                return int(token)
    except:
        return np.nan
    return np.nan

def main():
    con = duckdb.connect(str(DB))
    df = con.sql("SELECT * FROM apps").to_df()

    # tip dönüşümleri
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["positive"] = pd.to_numeric(df["positive"], errors="coerce")
    df["negative"] = pd.to_numeric(df["negative"], errors="coerce")

    # release_year
    df["release_year"] = df["release_date"].apply(parse_year)
    df = df[df["release_year"].between(MIN_YEAR, datetime.now().year, inclusive="both")]

    # uç değer filtresi
    df = df[(df["price"].isna()) | ((df["price"] >= 0) & (df["price"] <= MAX_PRICE_EUR))]

    # oranlar
    total_reviews = (df["positive"].fillna(0) + df["negative"].fillna(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["positive"].fillna(0) / total_reviews.replace(0, np.nan)
    df["positive_ratio"] = ratio.clip(0, 1)
    df["log_review_count"] = np.log1p(total_reviews)

    # yaş
    df["age_years"] = datetime.now().year - df["release_year"]

    # NaN name drop
    df = df[~df["name"].isna()]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    log.info(f"Cleaned {len(df)} rows → {OUT_PARQUET}")

    con.close()

if __name__ == "__main__":
    main()
