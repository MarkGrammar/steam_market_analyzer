# src/features/build_embeddings.py
from pathlib import Path
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss

# Girdiler
DB_PARQUET = Path("data/processed/apps_clean.parquet")
TAGS_CSV   = Path("data/processed/tags.csv")
GENRES_CSV = Path("data/processed/genres.csv")

# Çıktılar
OUT_DIR    = Path("data/processed/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_NPY    = OUT_DIR / "embeddings.npy"        # (n, d)
IDS_NPY    = OUT_DIR / "app_ids.npy"           # (n,)
MAP_CSV    = OUT_DIR / "index_mapping.csv"     # app_id, name, price, positive_ratio, release_year
INDEX_FAI  = OUT_DIR / "faiss.index"           # FAISS FlatIP index
TEXTS_PARQ = OUT_DIR / "text_features.parquet" # app_id, name, text_features (isteğe bağlı)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_base_table() -> pd.DataFrame:
    """apps_clean.parquet'ten gerekli kolonları alır; duplikeleri temizler."""
    df = pd.read_parquet(DB_PARQUET)
    keep = ["app_id", "name", "price", "positive_ratio", "release_year"]
    df = df[keep].dropna(subset=["app_id", "name"]).drop_duplicates(subset=["app_id"])
    df["app_id"] = df["app_id"].astype("int64")
    return df

def load_agg(path: Path, out_col: str) -> pd.DataFrame:
    """
    tags.csv veya genres.csv gibi dosyalarda app_id bazında benzersiz birleşim döndürür.
    CSV başlıklarının ['app_id', '<col>'] gibi olduğunu varsayar.
    """
    if not path.exists():
        return pd.DataFrame(columns=["app_id", out_col])
    df = pd.read_csv(path)
    if df.empty or "app_id" not in df.columns:
        return pd.DataFrame(columns=["app_id", out_col])

    # app_id + tek kolon olsun
    other_cols = [c for c in df.columns if c != "app_id"]
    if not other_cols:
        return pd.DataFrame(columns=["app_id", out_col])
    col = other_cols[0]

    grp = (
        df.dropna(subset=["app_id", col])
          .assign(app_id=lambda x: x["app_id"].astype("int64"))
          .groupby("app_id")[col]
          .apply(lambda s: " ".join(sorted(set(map(str, s)))))
          .reset_index()
          .rename(columns={col: out_col})
    )
    return grp

def build_text_features(df_apps: pd.DataFrame,
                        df_tags: pd.DataFrame,
                        df_genres: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_apps
        .merge(df_tags,   on="app_id", how="left")
        .merge(df_genres, on="app_id", how="left")
        .fillna({"name": "", "tag": "", "genre": ""})
    )
    # name + tags + genres (lower)
    df["text_features"] = (
        (df["name"].astype(str) + " " +
         df.get("tag",   "").astype(str) + " " +
         df.get("genre", "").astype(str))
        .str.strip()
        .str.lower()
    )
    # boş textler hariç
    df = df[df["text_features"].str.len() > 0].reset_index(drop=True)
    # deterministik sıralama (opsiyonel)
    df = df.sort_values("app_id").reset_index(drop=True)
    return df

def embed_texts(texts, model_name: str = MODEL_NAME, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # kendimiz normalize edeceğiz
    ).astype("float32")
    # cosine = dot için L2 normalize
    faiss.normalize_L2(embs)
    return embs

def build_faiss_index(embs: np.ndarray) -> faiss.Index:
    d = int(embs.shape[1])
    index = faiss.IndexFlatIP(d)  # normalized vektörlerle cosine
    index.add(embs)
    return index

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print(">> Loading tables ...")
    df_apps   = load_base_table()
    df_tags   = load_agg(TAGS_CSV,   "tag")
    df_genres = load_agg(GENRES_CSV, "genre")

    print(">> Building text features ...")
    df = build_text_features(df_apps, df_tags, df_genres)
    n = len(df)
    if n == 0:
        raise SystemExit("No rows to embed. Check apps_clean.parquet / tags.csv / genres.csv")

    print(f">> Encoding {n} items with {MODEL_NAME} ...")
    embs = embed_texts(df["text_features"].tolist())

    if embs.shape[0] != n:
        raise RuntimeError(f"Embedding count mismatch: {embs.shape[0]} vs {n}")

    print(">> Saving artifacts ...")
    app_ids = df["app_id"].to_numpy(dtype="int64")
    np.save(EMB_NPY, embs)
    np.save(IDS_NPY, app_ids)

    # UI ve debug için faydalı dosyalar
    df_out = df[["app_id", "name", "price", "positive_ratio", "release_year", "text_features"]].copy()
    df_out.to_parquet(TEXTS_PARQ, index=False)
    df_out.drop(columns=["text_features"]).to_csv(MAP_CSV, index=False)

    print(">> Building FAISS index ...")
    index = build_faiss_index(embs)
    faiss.write_index(index, str(INDEX_FAI))

    print(f"Done ✅  n={n}, dim={embs.shape[1]}")
    print(f"- {EMB_NPY}")
    print(f"- {IDS_NPY}")
    print(f"- {INDEX_FAI}")
    print(f"- {MAP_CSV}")
    print(f"- {TEXTS_PARQ}")

if __name__ == "__main__":
    main()
