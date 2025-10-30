# src/search/utils.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

EMB_DIR    = Path("data/processed/embeddings")
EMB_NPY    = EMB_DIR / "embeddings.npy"
IDS_NPY    = EMB_DIR / "app_ids.npy"
MAP_CSV    = EMB_DIR / "index_mapping.csv"
INDEX_FAI  = EMB_DIR / "faiss.index"
TEXTS_PARQ = EMB_DIR / "text_features.parquet"
TAGS_CSV   = Path("data/processed/tags.csv")
GENRES_CSV = Path("data/processed/genres.csv")

# ---- Loaders ----
def load_faiss_and_mapping():
    missing = [p for p in [EMB_NPY, IDS_NPY, MAP_CSV, INDEX_FAI, TEXTS_PARQ] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {', '.join(map(str, missing))}. Run `make embed`.")
    embs = np.load(EMB_NPY)
    app_ids = np.load(IDS_NPY)
    mapping = pd.read_csv(MAP_CSV)
    texts = pd.read_parquet(TEXTS_PARQ)
    index = faiss.read_index(str(INDEX_FAI))
    return embs, app_ids, mapping, texts, index

# src/search/utils.py içinde:
def load_tfidf():
    from joblib import load as joblib_load
    vec_path = EMB_DIR / "tfidf_vectorizer.pkl"
    mat_path = EMB_DIR / "tfidf_matrix.npz"
    map_csv  = MAP_CSV
    if not vec_path.exists() or not mat_path.exists() or not map_csv.exists():
        raise FileNotFoundError("TF-IDF artifacts missing. Run `make tfidf`.")
    try:
        vectorizer = joblib_load(vec_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {vec_path}. Rebuild with `make tfidf`.") from e
    matrix = load_npz(mat_path)
    mapping = pd.read_csv(map_csv)
    return vectorizer, matrix, mapping

def load_taxonomy():
    tdf = pd.DataFrame(columns=["app_id","tags_str"])
    gdf = pd.DataFrame(columns=["app_id","genres_str"])
    if TAGS_CSV.exists():
        t = pd.read_csv(TAGS_CSV)
        if "tag" in t.columns and not t.empty:
            tdf = (t.dropna().groupby("app_id")["tag"]
                     .apply(lambda s: ", ".join(sorted(set(map(str, s)))))
                     .reset_index().rename(columns={"tag":"tags_str"}))
    if GENRES_CSV.exists():
        g = pd.read_csv(GENRES_CSV)
        if "genre" in g.columns and not g.empty:
            gdf = (g.dropna().groupby("app_id")["genre"]
                     .apply(lambda s: ", ".join(sorted(set(map(str, s)))))
                     .reset_index().rename(columns={"genre":"genres_str"}))
    return tdf, gdf

def enrich_with_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    tdf, gdf = load_taxonomy()
    return df.merge(tdf, on="app_id", how="left").merge(gdf, on="app_id", how="left")

# ---- Math helpers ----
def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    faiss.normalize_L2(x)
    return x

def minmax01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)

def jaccard(a_set: set[str], b_set: set[str]) -> float:
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / max(union, 1)

def get_tagset(row) -> set[str]:
    tags = (row.get("tags_str") or "")
    genres = (row.get("genres_str") or "")
    pool = []
    if tags: pool += [t.strip().lower() for t in tags.split(",")]
    if genres: pool += [g.strip().lower() for g in genres.split(",")]
    return set(filter(None, pool))
