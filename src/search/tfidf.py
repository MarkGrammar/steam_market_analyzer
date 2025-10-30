# src/search/tfidf.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

EMB_DIR = Path("data/processed/embeddings")
VEC_PKL = EMB_DIR / "tfidf_vectorizer.pkl"
MATRIX  = EMB_DIR / "tfidf_matrix.npz"
ID_CSV  = EMB_DIR / "index_mapping.csv"   # app_id sırası

def _abs(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

def _load():
    # Dosyalar var mı?
    missing = [p for p in [VEC_PKL, MATRIX, ID_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "TF-IDF artifacts missing. Run `make tfidf`.\n"
            + "\n".join(f"- missing: {_abs(p)}" for p in missing)
        )

    # Vektörizeri joblib ile yükle (pickle yerine)
    try:
        vec = joblib.load(VEC_PKL)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TF-IDF vectorizer with joblib: {_abs(VEC_PKL)}\n{e}"
        ) from e

    # Matris ve mapping
    try:
        X = load_npz(MATRIX)
    except Exception as e:
        raise RuntimeError(f"Failed to load TF-IDF matrix: {_abs(MATRIX)}\n{e}") from e

    try:
        mapping = pd.read_csv(ID_CSV)
    except Exception as e:
        raise RuntimeError(f"Failed to load index mapping: {_abs(ID_CSV)}\n{e}") from e

    return vec, X, mapping

def tfidf_search(query: str, topk: int = 500) -> pd.DataFrame:
    """
    Serbest metin query → TF-IDF cosine benzerlik skoru.
    Dönen kolonlar: ['app_id','score'] (büyük = daha iyi).
    """
    vec, X, mapping = _load()
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["app_id", "score"])

    # Query vektörü
    q_vec = vec.transform([q])
    sims = cosine_similarity(q_vec, X).ravel()

    # En iyi k
    k = int(min(max(1, topk), sims.shape[0]))
    idx = np.argpartition(-sims, k - 1)[:k]
    # Skora göre sırala
    idx = idx[np.argsort(-sims[idx])]

    out = pd.DataFrame({
        "app_id": mapping.iloc[idx]["app_id"].astype(np.int64).values,
        "score": sims[idx].astype(float),
    })

    return out[["app_id", "score"]]
