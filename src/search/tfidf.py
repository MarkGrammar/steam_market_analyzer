# src/search/tfidf.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


def _repo_root() -> Path:
    # .../src/search/tfidf.py  -> parents[2]  = repo root (.. / .. / ..)
    return Path(__file__).resolve().parents[2]


def _candidates():
    """TF-IDF artifactları için olası klasörler (öncelik sırasıyla)."""
    root = _repo_root()
    return [
        root / "data" / "processed" / "embeddings",
        Path.cwd() / "data" / "processed" / "embeddings",
    ]


def _load():
    """Bulursa (vectorizer, X, mapping) döner; bulamazsa (None, None, None)."""
    vec_path = mat_path = map_path = None
    for base in _candidates():
        v = base / "tfidf_vectorizer.pkl"
        m = base / "tfidf_matrix.npz"
        mp = base / "index_mapping.csv"
        if v.exists() and m.exists() and mp.exists():
            vec_path, mat_path, map_path = v, m, mp
            break

    if not (vec_path and mat_path and map_path):
        return None, None, None

    try:
        vec = joblib.load(vec_path)
        X = load_npz(mat_path)
        mapping = pd.read_csv(map_path)
        # app_id kolonunun int olduğundan emin olalım
        if "app_id" in mapping.columns:
            mapping["app_id"] = mapping["app_id"].astype("int64", errors="ignore")
        return vec, X, mapping
    except Exception:
        # Herhangi bir yükleme hatasında güvenli şekilde None döndür.
        return None, None, None


def tfidf_search(query: str, topk: int = 500) -> pd.DataFrame:
    """
    Serbest metin query → TF-IDF cosine skoru.
    Dönüş: DataFrame(['app_id','score']) – büyük skor = daha iyi.
    Artifact yoksa veya query boşsa boş DataFrame döner.
    """
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["app_id", "score"])

    vec, X, mapping = _load()
    if vec is None or X is None or mapping is None or "app_id" not in mapping.columns:
        # Artifact bulunamadı → kırmadan boş dön.
        return pd.DataFrame(columns=["app_id", "score"])

    # Sorguyu vektörize et
    q_vec = vec.transform([q])
    sims = cosine_similarity(q_vec, X).ravel()

    k = int(min(max(1, topk), sims.shape[0]))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    out = pd.DataFrame(
        {
            "app_id": mapping.iloc[idx]["app_id"].astype(np.int64).values,
            "score": sims[idx].astype(float),
        }
    )
    return out[["app_id", "score"]]
