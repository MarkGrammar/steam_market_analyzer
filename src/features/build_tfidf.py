# src/features/build_tfidf.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp

EMB_DIR     = Path("data/processed/embeddings")
TEXTS_PARQ  = EMB_DIR / "text_features.parquet"
VEC_PKL     = EMB_DIR / "tfidf_vectorizer.pkl"
MATRIX_NPZ  = EMB_DIR / "tfidf_matrix.npz"
ID_MAP_CSV  = EMB_DIR / "index_mapping.csv"  # app_id sırası için

def normalize_space(s: str) -> str:
    return " ".join((s or "").lower().split())

def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    if not TEXTS_PARQ.exists():
        raise FileNotFoundError(f"Missing: {TEXTS_PARQ}. Run: make embed")

    df = pd.read_parquet(TEXTS_PARQ)
    if "text_features" not in df.columns:
        raise ValueError("text_features column missing")

    # FAISS mapping ile aynı sırada olduğumuzdan emin olalım
    map_df = pd.read_csv(ID_MAP_CSV)
    df = df.merge(map_df[["app_id"]], on="app_id", how="right")
    texts = df["text_features"].fillna("").map(normalize_space).tolist()

    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.6,
        strip_accents="unicode",
        lowercase=False,
    )
    X = vec.fit_transform(texts)  # csr_matrix

    joblib.dump(vec, VEC_PKL)
    sp.save_npz(MATRIX_NPZ, X)

    print(f"Saved TF-IDF vectorizer → {VEC_PKL}")
    print(f"Saved TF-IDF matrix    → {MATRIX_NPZ} shape={X.shape}")

if __name__ == "__main__":
    main()
