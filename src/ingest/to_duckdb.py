
# src/ingest/to_duckdb.py
from pathlib import Path
import duckdb
from src.utils.logging import get_logger

log = get_logger("to_duckdb")

DB_PATH = Path("data/processed/steam.duckdb")
APPS = "data/processed/apps.csv"
TAGS = "data/processed/tags.csv"
GENRES = "data/processed/genres.csv"

DDL = """
CREATE OR REPLACE TABLE apps AS SELECT * FROM read_csv_auto('{apps}', IGNORE_ERRORS=TRUE);
CREATE OR REPLACE TABLE tags AS SELECT * FROM read_csv_auto('{tags}', IGNORE_ERRORS=TRUE);
CREATE OR REPLACE TABLE genres AS SELECT * FROM read_csv_auto('{genres}', IGNORE_ERRORS=TRUE);

-- yardımcı indexler
CREATE INDEX IF NOT EXISTS idx_apps_id ON apps(app_id);
CREATE INDEX IF NOT EXISTS idx_tags_id ON tags(app_id);
CREATE INDEX IF NOT EXISTS idx_genres_id ON genres(app_id);
"""

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(DDL.format(apps=APPS, tags=TAGS, genres=GENRES))
    n_apps = con.sql("SELECT COUNT(*) FROM apps").fetchone()[0]
    log.info(f"DuckDB ready at {DB_PATH} — apps rows: {n_apps}")
    con.close()

if __name__ == "__main__":
    main()
