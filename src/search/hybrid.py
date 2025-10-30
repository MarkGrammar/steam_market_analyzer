# src/search/hybrid.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Yerel modüller (UI import etme → circular import kaçın)
from .tfidf import tfidf_search          # -> DataFrame['app_id','score']
from .dense import dense_search          # -> DataFrame['app_id','score']

EMB_DIR = Path("data/processed/embeddings")
MAP_CSV = EMB_DIR / "index_mapping.csv"

def _load_mapping() -> pd.DataFrame:
    if not MAP_CSV.exists():
        raise FileNotFoundError(f"Missing mapping: {MAP_CSV}. Run `make embed` first.")
    return pd.read_csv(MAP_CSV)

def _ensure_query_text(query_fields: str | dict | None) -> str:
    if query_fields is None:
        return ""
    if isinstance(query_fields, str):
        return query_fields
    if isinstance(query_fields, dict):
        parts = []
        for k in ["name", "tags", "genres", "desc", "description"]:
            v = query_fields.get(k)
            if v:
                parts.append(str(v))
        for k, v in query_fields.items():
            if k in {"name", "tags", "genres", "desc", "description"}:
                continue
            if v:
                parts.append(str(v))
        return " ".join(parts).strip()
    return str(query_fields)

def _coerce_score_df(df: pd.DataFrame | None, source: str) -> pd.DataFrame:
    """Her tür dataframe'i ['app_id','score'] standardına çevir."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["app_id", "score"])
    tmp = df.copy()
    # app_id adı
    cols_lower = {c.lower(): c for c in tmp.columns}
    if "app_id" not in tmp.columns:
        if "appid" in cols_lower:
            tmp = tmp.rename(columns={cols_lower["appid"]: "app_id"})
        else:
            raise ValueError(f"{source}: missing 'app_id' column (got {list(tmp.columns)})")
    # score adı
    if "score" not in tmp.columns:
        for cand in ("similarity", "cosine", "sim", "rrf"):
            if cand in tmp.columns:
                tmp = tmp.rename(columns={cand: "score"})
                break
            if cand in cols_lower:
                tmp = tmp.rename(columns={cols_lower[cand]: "score"})
                break
        if "score" not in tmp.columns:
            if "rank" in tmp.columns:
                tmp["score"] = 1.0 / (60.0 + pd.to_numeric(tmp["rank"], errors="coerce"))
            else:
                # hepsi başarısızsa dummy skor
                tmp["score"] = 1.0
    tmp = tmp[["app_id", "score"]].dropna()
    tmp["app_id"] = pd.to_numeric(tmp["app_id"], errors="coerce").dropna().astype(np.int64)
    tmp["score"]  = pd.to_numeric(tmp["score"], errors="coerce").fillna(0.0).astype(float)
    # Aynı app_id birden çok kez geldiyse en yüksek skoru al
    tmp = tmp.groupby("app_id", as_index=False)["score"].max()
    return tmp

def rrf_fuse(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    """Reciprocal Rank Fusion."""
    def _ranks(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0 or "score" not in df.columns:
            return pd.DataFrame(columns=["app_id","rank"])
        tmp = df[["app_id","score"]].dropna(subset=["app_id"]).copy()
        tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp)+1)
        return tmp[["app_id","rank"]]

    ra = _ranks(df_a)
    rb = _ranks(df_b)
    if len(ra) == 0 and len(rb) == 0:
        return pd.DataFrame(columns=["app_id", "rrf"])

    all_ids = pd.DataFrame({"app_id": pd.unique(pd.concat([ra["app_id"], rb["app_id"]], ignore_index=True))})
    all_ids = all_ids.merge(ra, on="app_id", how="left").merge(rb, on="app_id", how="left", suffixes=("_a", "_b"))
    all_ids["rank_a"] = all_ids["rank_a"].fillna(1e6)
    all_ids["rank_b"] = all_ids["rank_b"].fillna(1e6)
    all_ids["rrf"] = 1.0 / (k + all_ids["rank_a"]) + 1.0 / (k + all_ids["rank_b"])
    return all_ids.sort_values("rrf", ascending=False).reset_index(drop=True)[["app_id", "rrf"]]

def search_hybrid_text(
    query_fields: str | dict,
    k_return: int = 400,
    k_each: int = 800,
    rrf_k: int = 60,
    alpha: float | None = None,  # backward-compat; IGNORED
) -> pd.DataFrame:
    """
    1) Dense ve TF-IDF sonuçlarını al (çıktılar ['app_id','score'])
    2) RRF fuse
    3) Mapping ile join → 'similarity' = rrf
    """
    q_text = _ensure_query_text(query_fields)
    d_dense_raw = dense_search(query_fields, topk=int(k_each))
    d_tfidf_raw = tfidf_search(q_text, topk=int(k_each))

    d_dense = _coerce_score_df(d_dense_raw, "dense_search")
    d_tfidf = _coerce_score_df(d_tfidf_raw, "tfidf_search")

    # Tek taraf boşsa fallback: boş olmayanı döndür (similarity=score)
    if len(d_dense) == 0 and len(d_tfidf) == 0:
        return pd.DataFrame(columns=["app_id", "similarity"])
    if len(d_dense) == 0:
        single = d_tfidf.sort_values("score", ascending=False).head(int(k_return)).copy()
        single = single.rename(columns={"score": "similarity"})
        mapping = _load_mapping()
        return single.merge(mapping, on="app_id", how="left").reset_index(drop=True)
    if len(d_tfidf) == 0:
        single = d_dense.sort_values("score", ascending=False).head(int(k_return)).copy()
        single = single.rename(columns={"score": "similarity"})
        mapping = _load_mapping()
        return single.merge(mapping, on="app_id", how="left").reset_index(drop=True)

    fused = rrf_fuse(d_dense, d_tfidf, k=int(rrf_k)).head(int(k_return))
    mapping = _load_mapping()
    out = fused.merge(mapping, on="app_id", how="left").rename(columns={"rrf": "similarity"})
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)
