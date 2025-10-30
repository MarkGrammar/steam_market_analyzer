# src/search/dense.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMB_DIR    = Path("data/processed/embeddings")
EMB_NPY    = EMB_DIR / "embeddings.npy"
IDS_NPY    = EMB_DIR / "app_ids.npy"
INDEX_FAI  = EMB_DIR / "faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_index = None
_model = None
_app_ids = None

def _ensure_query_text(fields) -> str:
    if isinstance(fields, str):
        return fields
    if isinstance(fields, dict):
        parts = []
        for k in ["name", "tags", "genres", "desc", "description"]:
            v = fields.get(k)
            if v:
                parts.append(str(v))
        for k, v in fields.items():
            if k in {"name", "tags", "genres", "desc", "description"}:
                continue
            if v:
                parts.append(str(v))
        return " ".join(parts).strip()
    return str(fields or "")

def _load():
    global _index, _model, _app_ids
    if _index is None:
        _index = faiss.read_index(str(INDEX_FAI))
        _app_ids = np.load(IDS_NPY)
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _index, _model, _app_ids

def dense_search(query_fields: dict | str | None, topk: int = 500) -> pd.DataFrame:
    """
    FAISS + sentence-transformers embedding araması.
    Her durumda ['app_id','score'] kolonlarını döndürür.
    score: (inner product / cosine benzeri) büyük = daha iyi.
    """
    try:
        index, model, app_ids = _load()
    except Exception:
        # Arifakt yoksa bile şıkça boş kolonlar döndür.
        return pd.DataFrame(columns=["app_id", "score"])

    # Tek metne indir
    q = _ensure_query_text(query_fields)
    if not q or not str(q).strip():
        return pd.DataFrame(columns=["app_id", "score"])

    # Query vektörü
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    # IP indeksinde düzgün skor için normalize edelim (cosine-vari)
    import faiss  # local import; modül mevcutsa
    faiss.normalize_L2(qv)

    # K sınırı ve boş indeks koruması
    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal <= 0:
        return pd.DataFrame(columns=["app_id", "score"])
    k = int(min(max(1, topk), ntotal))

    # FAISS araması
    sims, idxs = index.search(qv, k)   # sims: [1,k], idxs: [1,k]
    sims = sims[0]
    idxs = idxs[0]

    # FAISS yeterli komşu bulamazsa -1 dönebilir → filtrele
    mask = idxs >= 0
    if not np.any(mask):
        return pd.DataFrame(columns=["app_id", "score"])

    idxs = idxs[mask]
    sims = sims[mask]

    # app_ids hizası; alınan dilim boşsa yine kolonlu boş df dön
    if len(idxs) == 0:
        return pd.DataFrame(columns=["app_id", "score"])

    # Çıktı DF
    out = pd.DataFrame({
        "app_id": app_ids[idxs].astype(np.int64, copy=False),
        "score": sims.astype(float, copy=False),
    })

    # Güvenlik: NaN/inf temizle ve skor sıralı olsun
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["app_id", "score"])
    if len(out) == 0:
        return pd.DataFrame(columns=["app_id", "score"])

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out[["app_id", "score"]]
