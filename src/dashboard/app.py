# src/dashboard/app.py

# --- make project root importable (so `import src.*` works) ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------------------



from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import altair as alt
import json as _json
import joblib
from dotenv import load_dotenv

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Bizim modüller
from src.features import advisor as _advisor
from src.search.hybrid import search_hybrid_text
from src.llm_clients.groq_client import groq_client  # LLM çağrısı
from src.search.field_embed import build_query_text_fields

from src.analytics.stats import (
    cohort_summary, price_bands, bootstrap_ci_mean, ratio_vs_price_year, add_sales_proxies, sales_regression, price_response_curve, heat_price_ratio_sales,
)




# ---- Sales metadata loader (review_count üretimi) ----
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def load_reviews_catalog() -> pd.DataFrame:
    """
    app_id bazında review_count (ve poz/neg/price) döndürür.
    Kaynak önceliği: apps_clean.parquet > apps.csv > steam.duckdb
    """
    base = Path("data/processed")
    df = None

    # 1) Parquet
    pq = base / "apps_clean.parquet"
    if pq.exists():
        try:
            tmp = pd.read_parquet(pq)
            # Sık görülen kolon adlarını normalize et
            cols = {c.lower(): c for c in tmp.columns}
            app_col = cols.get("app_id") or cols.get("appid") or "app_id"
            # review_count yoksa positive+negative ile üret
            if not {"review_count", "reviews_total"}.intersection({c.lower() for c in tmp.columns}):
                pos = pd.to_numeric(tmp.get("positive"), errors="coerce")
                neg = pd.to_numeric(tmp.get("negative"), errors="coerce")
                tmp["review_count"] = (pos.fillna(0) + neg.fillna(0)).astype("Int64")
            elif "reviews_total" in tmp.columns:
                tmp["review_count"] = pd.to_numeric(tmp["reviews_total"], errors="coerce").astype("Int64")

            df = tmp.rename(columns={app_col: "app_id"})[["app_id","review_count"]].copy()
            # Varsa faydalı alanları da bırak
            for extra in ("positive","negative","price","price_final","price_eur"):
                if extra in tmp.columns and extra not in df.columns:
                    df[extra] = tmp[extra]
        except Exception:
            df = None

    # 2) CSV fallback
    if df is None:
        csv = base / "apps.csv"
        if csv.exists():
            tmp = pd.read_csv(csv)
            # app_id / review_count üret
            app_col = "app_id" if "app_id" in tmp.columns else ("appid" if "appid" in tmp.columns else None)
            if app_col:
                if "review_count" not in tmp.columns:
                    pos = pd.to_numeric(tmp.get("positive"), errors="coerce")
                    neg = pd.to_numeric(tmp.get("negative"), errors="coerce")
                    tmp["review_count"] = (pos.fillna(0) + neg.fillna(0)).astype("Int64")
                df = tmp.rename(columns={app_col: "app_id"})[["app_id","review_count"]].copy()
                for extra in ("positive","negative","price","price_final","price_eur"):
                    if extra in tmp.columns and extra not in df.columns:
                        df[extra] = tmp[extra]

    # 3) DuckDB fallback
    if df is None:
        db = base / "steam.duckdb"
        if db.exists():
            try:
                import duckdb
                con = duckdb.connect(str(db))
                # tablo adınız farklıysa değiştirin (ör: apps, games, steam_apps…)
                q = """
                SELECT
                  CAST(app_id AS BIGINT) AS app_id,
                  COALESCE(review_count, positive + negative) AS review_count,
                  positive, negative, price
                FROM apps
                """
                df = con.execute(q).df()
                con.close()
            except Exception:
                df = None

    if df is None:
        # hiçbir kaynak yoksa boş dön
        return pd.DataFrame(columns=["app_id","review_count"])

    # Türleri düzelt
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    return df
    
    
def add_sales_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'e review_count, sales_proxy, revenue_proxy ekler."""
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    meta = load_reviews_catalog()  # app_id, review_count, (+ price/pos/neg)

    out = out.merge(meta, on="app_id", how="left", suffixes=("", "_meta"))

    # Review count yoksa bir kez daha üretmeyi dene (poz+neg'ten)
    if "review_count" not in out.columns or out["review_count"].isna().all():
        pos = pd.to_numeric(out.get("positive"), errors="coerce").fillna(0)
        neg = pd.to_numeric(out.get("negative"), errors="coerce").fillna(0)
        out["review_count"] = (pos + neg).astype("Int64")

    # Fiyat alanı seçimi: mevcut price, yoksa meta'dan
    price = pd.to_numeric(out.get("price"), errors="coerce")
    if price.isna().all():
        price = pd.to_numeric(out.get("price_final"), errors="coerce")
    if price.isna().all():
        price = pd.to_numeric(out.get("price_eur"), errors="coerce")
    if price.isna().all():
        price = pd.to_numeric(out.get("price_meta"), errors="coerce")
    price = price.fillna(0.0)

    rc = pd.to_numeric(out["review_count"], errors="coerce").fillna(0)

    out["sales_proxy"]   = rc
    out["revenue_proxy"] = rc * price
    return out

# --- .env yükle (repo kökü) ---
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Bridge: Streamlit secrets -> env
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# alternatif hızlı: "BAAI/bge-small-en-v1.5" (normalize_embeddings=True önerilir)

# ----------------------------
# JSON helpers
# ----------------------------
def _json_safe(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, _pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, _pd.Series):
        return o.to_dict()
    if isinstance(o, (_np.integer, _np.int_, _np.int32, _np.int64)):
        return int(o)
    if isinstance(o, (_np.floating, _np.float_, _np.float32, _np.float64)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)
    return str(o)

# ----------------------------
# UI CONFIG & THEME
# ----------------------------
st.set_page_config(
    page_title="Game Concept Market Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global UI + Kart/başlık stilleri
st.markdown("""
<style>
:root{
  --bg:#0b0f17; --card:#121826; --muted:#a3b1c6; --text:#e6edf7;
  --brand:#6aa3ff; --accent:#22d3ee; --chip:#1b2537; --chip-border:#243045;
  --success:#22c55e;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--text) !important; }
h1,h2,h3,h4{ color:var(--text) !important; }
small, .stCaption, p, label, .stMarkdown{ color:var(--muted) !important; }

/* Eski genel bloklar (tema) */
.st-emotion-cache-16idsys, .st-emotion-cache-1r4qj8v, .st-emotion-cache-1r6slb0{
  background:var(--bg) !important;
}
.block-card{
  background:var(--card); border:1px solid #1b2332; border-radius:14px;
  padding:18px 18px 14px; box-shadow:0 0 0 1px rgba(255,255,255,.02) inset, 0 10px 30px rgba(0,0,0,.35);
}
.grid{ display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.kpi{
  background:linear-gradient(180deg, rgba(34,211,238,.05), rgba(34,211,238,.02));
  border:1px solid #253045; padding:14px; border-radius:12px;
}
.kpi h3{ margin:0 0 6px; font-size:.85rem; color:var(--muted); }
.kpi .v{ font-size:1.35rem; font-weight:700; color:var(--text); }
.chip{
  display:inline-flex; align-items:center; gap:8px; padding:6px 10px; margin:3px 6px 0 0;
  border-radius:999px; background:var(--chip); border:1px solid var(--chip-border);
  font-size:12px; color:var(--muted);
}
.chip .dot{ width:8px; height:8px; border-radius:50%; background:var(--accent); display:inline-block; }
.sticky{
  position:sticky; top:0; z-index:20; background:linear-gradient(180deg, rgba(11,15,23,.85), rgba(11,15,23,.55));
  backdrop-filter:blur(8px); border-bottom:1px solid #1a2232; padding:10px 12px 6px; margin:-10px -8px 10px;
}
.btn-primary .st-emotion-cache-7ym5gk{
  background:var(--brand) !important; color:#0b0f17 !important; font-weight:600 !important;
  border-radius:10px !important; border:none !important;
}
.metric-ok{ color:var(--success); font-weight:600; }
.hr{ height:1px; background:#1a2232; width:100%; margin:12px 0 6px; }
.card-title a{ color:var(--brand); text-decoration:none; }

/* --- Kart + Başlık hizalama (yeni) --- */
.sma-card{
  background:#0e1217; border:1px solid rgba(255,255,255,.06);
  border-radius:16px; padding:16px 18px; margin:12px 0;
}
.sma-card-head{ display:flex; align-items:center; gap:12px; }

/* Oyun adı: tek satır + ellipsis */
.sma-title{
  font-weight:700; font-size:20px; line-height:1.25;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;
}

/* Sağdaki mağaza linki */
.sma-store{ color:#8ab4ff; text-decoration:none; font-weight:600; opacity:.95; }
.sma-store:hover{ text-decoration:underline; }

/* Meta bilgi satırı */
.sma-meta{ opacity:.85; margin-top:2px; font-size:14px; }

/* Similarity metni */
.sma-sim{ margin:10px 0 4px; font-size:14px; opacity:.9; }

/* Tag chipleri (render_cards içinde HTML ile basıyoruz) */
.sma-chip{
  display:inline-flex; align-items:center; background:#1b2330;
  border-radius:999px; padding:6px 12px; margin:6px 8px 0 0; font-size:14px; color:var(--muted);
}
</style>
""", unsafe_allow_html=True)
# ----------------------------
# UI state defaults (tek seferlik)
# ----------------------------
if "init_done" not in st.session_state:
    st.session_state.update({
        "page": "Market Explorer",
        # Global filters
        "topk": 100,
        "max_price": 60.0,
        "min_ratio": 0.70,
        "year_range": (2016, 2025),
        "min_sim": 0.0,
        "show_table": True,
        # Market Explorer
        "mx_selected_tags": [],
        "mx_results": None,
        # What-If
        "wi_selected_tags": [],
        "wi_price": 15.0,
        "wi_year": 2024,
        "wi_min_match": 1,
        "wi_team_size": 3,
        "wi_dev_months": 12,
        "wi_risk": "Medium",
        "wi_graphics": "2D",
        "wi_perspective": "Side Scroller",
        "wi_online_mode": "None",
        "wi_platforms": [],
        "wi_results": None,
        "advisor_report": None,
    })
    st.session_state["init_done"] = True
    
    
    
    


# ----------------------------
# Paths & constants
# ----------------------------
EMB_DIR    = Path("data/processed/embeddings")
EMB_NPY    = EMB_DIR / "embeddings.npy"
IDS_NPY    = EMB_DIR / "app_ids.npy"
MAP_CSV    = EMB_DIR / "index_mapping.csv"
INDEX_FAI  = EMB_DIR / "faiss.index"
TEXTS_PARQ = EMB_DIR / "text_features.parquet"
APPS_CLEAN = Path("data/processed/apps_clean.parquet")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TAGS_CSV   = Path("data/processed/tags.csv")
GENRES_CSV = Path("data/processed/genres.csv")

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_tfidf():
    vec_path = Path("data/processed/embeddings/tfidf_vectorizer.pkl")
    mat_path = Path("data/processed/embeddings/tfidf_matrix.npz")
    map_csv  = Path("data/processed/embeddings/index_mapping.csv")

    missing = [p for p in [vec_path, mat_path, map_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "TF-IDF artifacts missing. Run `make tfidf` first.\n"
            + "\n".join(f"- missing: {p}" for p in missing)
        )

    try:
        vectorizer = joblib.load(vec_path)      # <-- pickle değil, joblib
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TF-IDF vectorizer ({vec_path}). "
            "It’s likely corrupted. Run `rm .../tfidf_vectorizer.pkl` then `make tfidf`."
        ) from e

    try:
        matrix = load_npz(mat_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TF-IDF matrix ({mat_path}). "
            "Delete it and re-run `make tfidf`."
        ) from e

    mapping = pd.read_csv(map_csv)
    return vectorizer, matrix, mapping

@st.cache_resource(show_spinner=False)
def load_model(name=MODEL_NAME):
    return SentenceTransformer(name)

def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    faiss.normalize_L2(x)
    return x

@st.cache_resource(show_spinner=False)
def load_tfidf():
    vec_path = EMB_DIR / "tfidf_vectorizer.pkl"
    mat_path = EMB_DIR / "tfidf_matrix.npz"
    map_csv  = EMB_DIR / "index_mapping.csv"
    for p in [vec_path, mat_path, map_csv]:
        if not p.exists():
            raise FileNotFoundError("TF-IDF artifacts missing. Run `make tfidf` first.")
    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TF-IDF vectorizer: {vec_path}. Rebuild with `make tfidf`."
        ) from e
    matrix = load_npz(mat_path)
    mapping = pd.read_csv(map_csv)
    return vectorizer, matrix, mapping

# ----------------------------
# Helpers (taxonomy / filters / cards)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_taxonomy():
    tdf = pd.DataFrame(columns=["app_id", "tags_str"])
    gdf = pd.DataFrame(columns=["app_id", "genres_str"])

    if TAGS_CSV.exists():
        t = pd.read_csv(TAGS_CSV)
        if not t.empty and "tag" in t.columns:
            tdf = (t.dropna()
                     .groupby("app_id")["tag"]
                     .apply(lambda s: ", ".join(sorted(set(map(str, s)))))
                     .reset_index()
                     .rename(columns={"tag": "tags_str"}))

    if GENRES_CSV.exists():
        g = pd.read_csv(GENRES_CSV)
        if not g.empty and "genre" in g.columns:
            gdf = (g.dropna()
                     .groupby("app_id")["genre"]
                     .apply(lambda s: ", ".join(sorted(set(map(str, s)))))
                     .reset_index()
                     .rename(columns={"genre": "genres_str"}))

    return tdf, gdf

def enrich_with_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    tdf, gdf = load_taxonomy()
    return df.merge(tdf, on="app_id", how="left").merge(gdf, on="app_id", how="left")

def apply_filters(df: pd.DataFrame, max_price: float | None, min_ratio: float | None,
                  year_min: int | None, year_max: int | None):
    out = df.copy()
    if max_price is not None and "price" in out.columns:
        out = out[(out["price"].isna()) | (out["price"] <= max_price)]
    if min_ratio is not None and "positive_ratio" in out.columns:
        out = out[out["positive_ratio"] >= min_ratio]
    if "release_year" in out.columns:
        if year_min is not None:
            out = out[out["release_year"] >= year_min]
        if year_max is not None:
            out = out[out["release_year"] <= year_max]
    return out

def _chip(text: str):
    st.markdown(f"""<span class="chip"><span class="dot"></span>{text}</span>""", unsafe_allow_html=True)

def render_cards(df: pd.DataFrame):
    for _, r in df.iterrows():
        name = r.get("name", "Unnamed")
        app_id = int(r["app_id"])
        year = int(r["release_year"]) if pd.notna(r.get("release_year")) else "—"
        price = f"€{r['price']}" if pd.notna(r.get('price')) else "Free/—"
        pr = f"{round(float(r['positive_ratio'])*100,1)}%" if pd.notna(r.get('positive_ratio')) else "—"
        sim = float(r.get("similarity", np.nan)) if pd.notna(r.get("similarity")) else np.nan

        # Başlık + link
        st.markdown(f"""
        <div class="sma-card">
          <div class="sma-card-head">
            <div class="sma-title">{name}</div>
            <a href="https://store.steampowered.com/app/{app_id}/" target="_blank" class="sma-store">Steam Store ↗</a>
          </div>
          <div class="sma-meta">ID: {app_id} • Year: {year} • Price: {price} • Positive ratio: {pr}</div>
        """, unsafe_allow_html=True)

        # Similarity bar
        if not np.isnan(sim):
            st.markdown(f'<div class="sma-sim">Similarity: {sim:.3f}</div>', unsafe_allow_html=True)
            st.progress(min(max(sim, 0.0), 1.0))

        # Tags + genres
        chips = []
        if r.get("genres_str"):
            chips += [g for g in r["genres_str"].split(", ")[:3]]
        if r.get("tags_str"):
            chips += [t for t in r["tags_str"].split(", ")[:4]]

        if chips:
            chip_html = "".join([f'<span class="sma-chip">{c}</span>' for c in chips])
            st.markdown(chip_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # sma-card kapat

# ---- Tile catalog (checkbox grid) ----
CANDIDATE_CATALOG = {
    "Genres": [
        "Indie","Action","Action Roguelike","RPG","Action RPG","Adventure",
        "Metroidvania","Souls-like","City Builder","Colony Sim","Simulation",
        "Strategy","Deckbuilding","Card Battler","Platformer"
    ],
    "Mechanics": [
        "Roguelike","Roguelite","Base Building","Crafting","Farming Sim",
        "Survival","Resource Management","Procedural Generation","Bullet Hell",
        "Permadeath","Boss Rush","Building","Automation"
    ],
    "Aesthetic": [
        "Pixel Graphics","Hand-drawn","2D","3D","Stylized","Cartoony",
        "Dark","Cute","Minimalist","Colorful"
    ],
    "Camera / Perspective": [
        "Side Scroller","Top-Down","Isometric","Third Person","First-Person"
    ],
    "Mode": [
        "Singleplayer","Co-op","Online Co-Op","Local Co-Op","PvP","MMO","Open World","Sandbox"
    ],
    "Scale": [
        "Short","Casual","Story Rich","Narrative","Replay Value"
    ],
}

import re

def _slug_key(s: str) -> str:
    """Make a safe, deterministic key chunk from any string."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip()).strip("_")

def render_tag_multipickers(catalog: dict, key_prefix: str) -> list[str]:
    """Kategori bazlı kare-kare checkbox grid."""
    picked = []
    for section, tags in catalog.items():
        st.markdown(f"###### {section}")
        cols = st.columns(4)
        for i, t in enumerate(tags):
            with cols[i % 4]:
                if st.checkbox(t, key=f"{key_prefix}_chk_{_slug_key(section)}_{_slug_key(t)}"):
                    picked.append(t)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    return sorted(set(picked))

def _tags_match_count(tags_str: str, must: list[str]) -> int:
    s = (tags_str or "").lower()
    return sum(1 for t in must if t.lower() in s)

def filter_by_tags_soft(df: pd.DataFrame, must_tags: list[str], min_match: int) -> pd.DataFrame:
    """tags_str içinde seçilen tag’lerden en az `min_match` kadarını içerenleri bırakır (soft-AND)."""
    if df is None or len(df) == 0 or not must_tags:
        return df
    tmp = df.copy()
    tmp["tag_match_count"] = tmp["tags_str"].fillna("").apply(lambda s: _tags_match_count(s, must_tags))
    tmp = tmp[tmp["tag_match_count"] >= int(min_match)]
    order_cols = [c for c in ["tag_match_count","similarity"] if c in tmp.columns]
    if order_cols:
        tmp = tmp.sort_values(order_cols, ascending=[False]*len(order_cols))
    return tmp.reset_index(drop=True)

def exclude_aaa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Çok büyük/AAA ölçekli yapımları kabaca ele.
    Kriterler:
      - Fiyat >= €39
      - Çok yüksek yorum hacmi (log_review_count > ~11 ≈ 60k+ yorum)
      - Ağır AAA etiketleri: Open World, MMO, Online PvP, Battle Royale
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # Fiyat
    price = pd.to_numeric(out.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0)
    mask_price = price >= 39.0

    # Yorum hacmi: varsa log_review_count kullan; yoksa positive+negative'tan türet
    if "log_review_count" in out.columns:
        log_rc = pd.to_numeric(out["log_review_count"], errors="coerce").fillna(0.0)
    else:
        pos = pd.to_numeric(out.get("positive", pd.Series(dtype=float)), errors="coerce").fillna(0)
        neg = pd.to_numeric(out.get("negative", pd.Series(dtype=float)), errors="coerce").fillna(0)
        log_rc = np.log1p(pos + neg)
        out["log_review_count"] = log_rc
    mask_reviews = log_rc > 11.0  # ~ 60k+ yorum

    # AAA etiketleri
    tags = out.get("tags_str", pd.Series(dtype=str)).fillna("").astype(str)
    mask_heavy = tags.str.contains(
        r"\b(Open World|Massively Multiplayer|Online PvP|Battle Royale)\b",
        case=False, regex=True
    )

    return out[~(mask_price | mask_reviews | mask_heavy)].reset_index(drop=True)

# ----------------------------
# Retrieval helpers
# ----------------------------
def search_by_text(query: str, k: int = 10):
    _, _, mapping, _, index = load_index_and_mapping()
    model = load_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    q = _norm(q)
    k = int(min(k, getattr(index, "ntotal", k)))
    sims, idxs = index.search(q, k)
    idxs = idxs[0].tolist()
    sims = sims[0].tolist()
    out = mapping.iloc[idxs].copy()
    out["similarity"] = [round(s, 4) for s in sims]
    return out

def search_by_app_id(app_id: int, k: int = 10):
    embs, app_ids, mapping, _, index = load_index_and_mapping()
    where = np.where(app_ids == int(app_id))[0]
    if len(where) == 0:
        raise ValueError(f"app_id not found in index: {app_id}")
    i = int(where[0])
    q = embs[i][None, :]
    k = int(min(k, getattr(index, "ntotal", k)))
    sims, idxs = index.search(q, k + 1)
    idxs = idxs[0].tolist()
    sims = sims[0].tolist()
    if idxs and idxs[0] == i:
        idxs, sims = idxs[1:], sims[1:]
    out = mapping.iloc[idxs].copy()
    out["similarity"] = [round(s, 4) for s in sims]
    return out



# ----------------------------
# HEADER
# ----------------------------
with st.container():
    st.markdown('<div class="sticky">', unsafe_allow_html=True)
    c1, c2 = st.columns([6,4])
    with c1:
        st.markdown("### 🎮 Game Concept Market Analyzer")
        st.caption("Hybrid retrieval • Modern UI • FAISS + TF-IDF • Cohort analytics • LLM advisor")
    with c2:
        # Index boyutu / mapping debug
        try:
            _, _, mapping_dbg, _, index_dbg = load_index_and_mapping()
            st.caption(f"Index: {getattr(index_dbg, 'ntotal', 'n/a')} • Items: {len(mapping_dbg)}")
        except Exception:
            st.caption("Index not loaded.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# SIDEBAR (global filters)
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Global Filters</div>', unsafe_allow_html=True)
    st.slider("Min similarity", 0.0, 1.0, key="min_sim")
    st.number_input("Max price (€)", min_value=0.0, step=1.0, key="max_price")
    st.slider("Min positive ratio", 0.0, 1.0, key="min_ratio")
    st.slider("Release year range", 2013, 2025, key="year_range")
    st.checkbox("Show table under cards", key="show_table")

# ----------------------------
# PAGES
# ----------------------------
tabs = st.tabs(["📊 Market Explorer", "🧪 What-If Simulator", "🤖 Advisor & Export"])
tab_mx, tab_wi, tab_adv = tabs

# ========= MARKET EXPLORER =========
with tab_mx:
    st.subheader("Belirli bir pazar dilimini inceleyin (tag seçerek)")
    left, right = st.columns([3, 1])

with left:
    picked_tags = render_tag_multipickers(CANDIDATE_CATALOG, "mx")
    st.session_state["mx_selected_tags"] = picked_tags
    st.info(f"Seçilen tag sayısı: {len(picked_tags)}")

    with right:
        st.markdown("#### Parametreler")
        min_match_default = max(1, min(3, len(picked_tags)//2 or 1))
        min_match = st.number_input(
            "Min tag overlap (soft-AND)",
            min_value=1,
            max_value=max(1, len(picked_tags) or 1),
            value=min_match_default,
            step=1,
            help="Seçtiğiniz tag’lerden en az kaç tanesi oyunda bulunmalı?",
            key="mx_min_match"
        )
        max_price  = st.session_state.get("max_price", 60.0)
        min_ratio  = st.session_state.get("min_ratio", 0.7)
        year_min, year_max = st.session_state.get("year_range", (2016, 2025))
        k_return = st.slider("Pool size (K)", 100, 1500, 600, 50,
                             help="Analiz öncesi toplanacak benzer oyun havuzu büyüklüğü")
        go_mx = st.button("Run Market Analysis", type="primary", use_container_width=True)

    if go_mx:
        if not picked_tags:
            st.warning("Lütfen en az 1 tag seçin.")
        else:
            query_text = " ".join(picked_tags)
            with st.spinner("Benzer oyunlar aranıyor (Hybrid)…"):
                try:
                    res = search_hybrid_text(
                        query_text,
                        k_return=k_return,
                        k_each=min(2*k_return, 1200),
                        alpha=0.65
                    )
                except Exception as e:
                    st.error(f"Aramada hata: {e}")
                    res = pd.DataFrame()

            if isinstance(res, pd.DataFrame) and not res.empty:
                res = enrich_with_taxonomy(res)
                res = filter_by_tags_soft(res, picked_tags, min_match)
                res = exclude_aaa(res)
                res = apply_filters(res, max_price, min_ratio, year_min, year_max)
                res = add_sales_proxies(res)              # <<< EKLE
                st.session_state["mx_results"] = res.copy()
            else:
                st.session_state["mx_results"] = pd.DataFrame()

    mx_df = st.session_state.get("mx_results")
    if mx_df is None or len(mx_df) == 0:
        st.info("Tag’leri seçip **Run Market Analysis**’e basın.")
    else:
        st.success(f"{len(mx_df)} eşleşme bulundu.")

        # --- Cohort Analytics (robust özet + CI + küçük regresyon) ---
        summary = cohort_summary(mx_df)
        ci_lo, ci_hi = bootstrap_ci_mean(pd.to_numeric(mx_df["positive_ratio"], errors="coerce"))
        reg = ratio_vs_price_year(mx_df)

        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            st.metric("Items", f'{summary["n_items"]}')
        with c1:
            st.metric("Median €", f'€{summary["price_median"]:.2f}' if summary["price_median"]==summary["price_median"] else "—")
        with c2:
            st.metric("MAD €", f'{summary["price_mad"]:.2f}' if summary["price_mad"]==summary["price_mad"] else "—")
        with c3:
            st.metric("Median ratio", f'{summary["ratio_median"]*100:.1f}%' if summary["ratio_median"]==summary["ratio_median"] else "—")
        with c4:
            if ci_lo==ci_lo and ci_hi==ci_hi:
                st.metric("Ratio mean CI95", f'{ci_lo*100:.1f}%–{ci_hi*100:.1f}%')
            else:
                st.metric("Ratio mean CI95", "—")

        # fiyat band histogramı (opsiyonel, mevcut “Price dist” sekmesine ek alternatif)
        px = pd.to_numeric(mx_df["price"], errors="coerce")
        bands_df = price_bands(px, step=2.0, max_price=float(st.session_state.get("max_price",60.0)))
        if len(bands_df):
            import altair as alt
            band_chart = alt.Chart(bands_df).mark_bar().encode(
                x=alt.X("band:N", sort=None, title="Price band"),
                y=alt.Y("count:Q", title="Count")
            )
            st.altair_chart(band_chart, use_container_width=True)

        # küçük regresyon bulgusu (yön ve R²’yi not olarak göster)
        if reg["beta_price"]==reg["beta_price"]:
            sign = "↓" if reg["beta_price"] < 0 else "↑"
            st.caption(f"Regression-lite: ratio ~ price + year → price coeff {reg['beta_price']:.3f} ({sign}), R²={reg['r2']:.2f}")
            
                # --- Advanced analytics (sales) — MARKET EXPLORER ---
        with st.expander("📊 Advanced analytics — sales & pricing", expanded=False):
            df_an = add_sales_proxies(mx_df)  # review_count varsa sales_proxy & revenue_proxy ekler

            if df_an is None or len(df_an) == 0 or "sales_proxy" not in df_an.columns:
                st.info("Bu panel için 'review_count' kolonu gerekli (sales proxy üretimi). Veri bulunamadı.")
            else:
                # 1) KPI'lar
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Median sales (proxy)", f"{int(np.nanmedian(df_an['sales_proxy'])):,}")
                with c2:
                    med_rev = np.nanmedian(df_an['revenue_proxy']) if 'revenue_proxy' in df_an.columns else np.nan
                    st.metric("Median revenue (proxy)", f"€{med_rev:.0f}" if pd.notna(med_rev) else "—")
                with c3:
                    st.metric("Items with sales", f"{(df_an['sales_proxy']>0).sum():,}")
                with c4:
                    st.metric("Conv. rate (assumed)", "7%")

                # 2) Regresyon
                reg = sales_regression(df_an)
                st.caption(
                    f"Model: log(sales) ~ log(price) + positive_ratio + year  |  "
                    f"n={reg['n']}  •  R²={reg['r2']:.2f}  •  price elasticity={reg['beta_price']:.2f}  "
                    f"•  ratio effect={reg['beta_ratio']:.2f}"
                )

                # 3) Scatter: log-price vs log-sales (+ trend)
                plot_df = df_an[["price","sales_proxy","name"]].replace([np.inf,-np.inf], np.nan).dropna()
                plot_df = plot_df[(plot_df["price"]>0) & (plot_df["sales_proxy"]>0)].copy()
                plot_df["log_price"] = np.log(plot_df["price"])
                plot_df["log_sales"] = np.log(plot_df["sales_proxy"])
                if len(plot_df) >= 5:
                    pts = alt.Chart(plot_df).mark_circle(opacity=0.7).encode(
                        x=alt.X("log_price:Q", title="log(Price)"),
                        y=alt.Y("log_sales:Q", title="log(Sales proxy)"),
                        tooltip=["name","price","sales_proxy"]
                    )
                    trend = pts.transform_regression("log_price","log_sales").mark_line()
                    st.altair_chart(pts + trend, use_container_width=True)

                # 4) Price response curve
                curve = price_response_curve(df_an)
                if len(curve) > 0:
                    st.markdown("**Price → expected sales (holding ratio/year at cohort means)**")
                    ch = alt.Chart(curve).mark_line().encode(
                        x=alt.X("price:Q", title="Price (€)"),
                        y=alt.Y("sales_pred:Q", title="Predicted sales (proxy)")
                    )
                    st.altair_chart(ch, use_container_width=True)

                # 5) Heatmap: price × ratio → median sales
                hm = heat_price_ratio_sales(df_an)
                if len(hm) > 0:
                    heat = alt.Chart(hm).mark_rect().encode(
                        x=alt.X("price_bin:N", title="Price bin"),
                        y=alt.Y("ratio_bin:N", title="Positive ratio bin"),
                        color=alt.Color("median_sales:Q", title="Median sales (proxy)", scale=alt.Scale(scheme="tealblues")),
                        tooltip=["price_bin","ratio_bin","median_sales"]
                    )
                    st.altair_chart(heat, use_container_width=True)

                # 6) Revenue histogram
                if "revenue_proxy" in df_an.columns and df_an["revenue_proxy"].notna().sum() > 0:
                    st.markdown("**Revenue proxy histogram**")
                    hist_df = df_an[["revenue_proxy"]].dropna()
                    hist = alt.Chart(hist_df).mark_bar().encode(
                        x=alt.X("revenue_proxy:Q", bin=alt.Bin(maxbins=30), title="Revenue proxy (€)"),
                        y=alt.Y("count()", title="Count")
                    )
                    st.altair_chart(hist, use_container_width=True)




        # KPI GRID
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            med_p = mx_df["price"].median(skipna=True)
            st.markdown('<div class="kpi"><h3>Median price</h3><div class="v">{}</div></div>'.format(
                f"€{med_p:.2f}" if pd.notna(med_p) else "—"), unsafe_allow_html=True)
        with c2:
            p25 = mx_df["price"].quantile(0.25)
            p75 = mx_df["price"].quantile(0.75)
            st.markdown('<div class="kpi"><h3>Price IQR</h3><div class="v">{}</div></div>'.format(
                f"€{p25:.0f}–€{p75:.0f}" if pd.notna(p25) and pd.notna(p75) else "—"), unsafe_allow_html=True)
        with c3:
            med_r = mx_df["positive_ratio"].median(skipna=True)
            st.markdown('<div class="kpi"><h3>Median positive ratio</h3><div class="v">{}</div></div>'.format(
                f"{med_r*100:.1f}%" if pd.notna(med_r) else "—"), unsafe_allow_html=True)


        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Charts in tabs
        t1, t2, t3 = st.tabs(["Scatter", "Price dist", "Positive ratio dist"])
        with t1:
            df_plot = mx_df[["name","price","positive_ratio","release_year"]].dropna()
            if len(df_plot) > 0:
                chart = (alt.Chart(df_plot)
                         .mark_circle(size=70, opacity=0.9)
                         .encode(
                             x=alt.X("price:Q", title="Price (€)"),
                             y=alt.Y("positive_ratio:Q", title="Positive ratio"),
                             tooltip=["name","price","positive_ratio","release_year"]
                         )
                         .interactive())
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Scatter için yeterli veri yok.")
        with t2:
            s = mx_df["price"].dropna()
            if len(s) > 0:
                bins = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150]
                cats = pd.cut(s.clip(lower=0, upper=bins[-1]), bins=bins, right=False, include_lowest=True)
                vc = cats.value_counts(sort=False)
                df_bins = pd.DataFrame({
                    "bin": [f"{int(iv.left)}–{int(iv.right)}" for iv in vc.index],
                    "count": vc.values
                })
                st.bar_chart(df_bins, x="bin", y="count")
            else:
                st.info("Fiyat verisi yetersiz.")
        with t3:
            s = mx_df["positive_ratio"].dropna().clip(0, 1)
            if len(s) > 0:
                bins = np.linspace(0, 1, 11)
                cats = pd.cut(s, bins=bins, right=False, include_lowest=True)
                vc = cats.value_counts(sort=False)

                def fmt(iv):
                    l = f"{iv.left:.1f}".rstrip('0').rstrip('.')
                    r = f"{iv.right:.1f}".rstrip('0').rstrip('.')
                    return f"{l}–{r}"

                df_bins = pd.DataFrame({
                    "bin": [fmt(iv) for iv in vc.index],
                    "count": vc.values
                })
                st.bar_chart(df_bins, x="bin", y="count")
            else:
                st.info("Positive ratio verisi yetersiz.")

        st.markdown("#### Examples (first 20)")
        render_cards(mx_df.head(20))
        if st.session_state["show_table"]:
            st.dataframe(mx_df.head(200), use_container_width=True, hide_index=True)

        csv_bytes = mx_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes,
                           file_name="market_explorer_results.csv", mime="text/csv", type="primary")

# ========= WHAT-IF =========
with tab_wi:
    st.subheader("Oyununuzu tarif edin (tag seçerek) ve üretim kapsamını girin")
    left, right = st.columns([3, 1])

    with left:
        selected_tags = render_tag_multipickers(CANDIDATE_CATALOG, "wi")
        st.session_state["wi_selected_tags"] = selected_tags
        st.info(f"Seçilen tag sayısı: {len(selected_tags)}")

    with right:
        st.markdown("#### Aday meta")
        price = st.number_input(
            "List price (€)", min_value=0.0,
            value=float(st.session_state.get("wi_price", 15.0)), step=0.5, key="wi_price_inp"
        )
        st.session_state["wi_price"] = price

        st.markdown("#### Production scope")
        team_size  = st.number_input("Team size (FTE)", min_value=1, max_value=50,
                                     value=st.session_state.get("wi_team_size", 3), step=1, key="wi_team_size")
        dev_months = st.number_input("Planned duration (months)", min_value=1, max_value=48,
                                     value=st.session_state.get("wi_dev_months", 12), step=1, key="wi_dev_months")
        st.selectbox("Production risk tolerance", ["Low","Medium","High"],
                     index=["Low","Medium","High"].index(st.session_state.get("wi_risk","Medium")),
                     key="wi_risk")

        st.markdown("#### Design / tech")
        st.selectbox("Graphics", ["2D","3D"],
                     index=["2D","3D"].index(st.session_state.get("wi_graphics","2D")),
                     key="wi_graphics")
        st.selectbox("Perspective", ["Side Scroller","Top-Down","Isometric","Third Person","First-Person"],
                     index=["Side Scroller","Top-Down","Isometric","Third Person","First-Person"].index(
                           st.session_state.get("wi_perspective","Side Scroller")),
                     key="wi_perspective")
        st.selectbox("Online mode", ["None","Co-op","PvP","MMO"],
                     index=["None","Co-op","PvP","MMO"].index(st.session_state.get("wi_online_mode","None")),
                     key="wi_online_mode")

        st.markdown("#### Platforms (optional)")
        st.multiselect("Target platforms", ["PC","Console","Mobile","Switch","PS","Xbox"],
                       default=st.session_state.get("wi_platforms", []), key="wi_platforms")

        min_match_default = max(1, min(4, len(selected_tags)//2 or 1))
        min_match = st.number_input(
            "Min tag overlap (soft-AND)",
            min_value=1,
            max_value=max(1, len(selected_tags) or 1),
            value=min_match_default,
            step=1,
            help="Seçtiğiniz tag’lerden en az kaç tanesi oyunda bulunmalı?",
            key="wi_min_match_inp"
        )
        st.session_state["wi_min_match"] = int(min_match)

        k = int(st.session_state.get("topk", 100))
        go_wi = st.button("Run What-If", type="primary", use_container_width=True, key="btn_run_wi")

    # --- Global filtre snapshot
    max_price  = st.session_state["max_price"]
    min_ratio  = st.session_state["min_ratio"]
    year_min, year_max = st.session_state["year_range"]

    # --- Arama
    if go_wi:
        if not selected_tags:
            st.warning("Lütfen en az 1–2 tag seçin.")
        else:
            query_fields = build_query_text_fields(
                selected_tags,
                perspective=st.session_state.get("wi_perspective"),
                graphics=st.session_state.get("wi_graphics")
            )

            with st.spinner("Benzer oyunlar aranıyor (Hybrid, field-aware)…"):
                try:
                    res = search_hybrid_text(
                        query_fields,
                        k_return=max(k, 200),
                        k_each=400,
                        rrf_k=60
                    )
                except Exception as e:
                    st.error(f"Hybrid aramada hata: {e}")
                    res = pd.DataFrame()

            if isinstance(res, pd.DataFrame) and not res.empty:
                res = enrich_with_taxonomy(res)
                res = filter_by_tags_soft(res, selected_tags, st.session_state["wi_min_match"])
                if not res.empty:
                    res = exclude_aaa(res)
                    res = apply_filters(res, max_price, min_ratio, year_min, year_max)
                    res = add_sales_proxies(res)  # <<< EKLE
                else:
                    st.warning("Bu kadar dar koşulla eşleşme bulunamadı. Eşiği düşürmeyi deneyin.")
                st.session_state["wi_results"] = res.copy()
            else:
                st.session_state["wi_results"] = pd.DataFrame()

    # --- Görüntüleme
    wi_df = st.session_state.get("wi_results")
    if wi_df is None or len(wi_df) == 0:
        st.info("Sol tarafta tag’leri seçip **Run What-If**’e basın.")
    else:
        st.success(f"{len(wi_df)} eşleşme bulundu.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="kpi"><h3>Matches</h3><div class="v">{}</div></div>'.format(len(wi_df)), unsafe_allow_html=True)
        with c2:
            med_p = wi_df["price"].median(skipna=True)
            st.markdown('<div class="kpi"><h3>Median price</h3><div class="v">{}</div></div>'.format(
                f"€{med_p:.2f}" if pd.notna(med_p) else "—"), unsafe_allow_html=True)
        with c3:
            med_r = wi_df["positive_ratio"].median(skipna=True)
            st.markdown('<div class="kpi"><h3>Median ratio</h3><div class="v">{}</div></div>'.format(
                f"{med_r*100:.1f}%" if pd.notna(med_r) else "—"), unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        sum_wi = cohort_summary(wi_df)
        cand_price = float(st.session_state.get("wi_price", 15.0))
        median_price = sum_wi["price_median"] if sum_wi["price_median"]==sum_wi["price_median"] else cand_price
        delta_vs_med = cand_price - median_price

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Cohort median €", f'€{median_price:.2f}' if median_price==median_price else "—")
        with d2:
            st.metric("Your price €", f'€{cand_price:.2f}')
        with d3:
            st.metric("Δ vs median", f'€{delta_vs_med:+.2f}')

        df_plot = wi_df[["name","price","positive_ratio","release_year"]].dropna()
        base = alt.Chart(df_plot).mark_circle(size=70, opacity=0.9).encode(
            x=alt.X("price:Q", title="Price (€)"),
            y=alt.Y("positive_ratio:Q", title="Positive ratio"),
            tooltip=["name", "price", "positive_ratio"]
        )
        cand_df = pd.DataFrame([{
            "name": "[CANDIDATE]",
            "price": float(st.session_state.get("wi_price", 15.0)),
            "positive_ratio": float(med_r) if pd.notna(med_r) else None,
        }])
        cand_mark = alt.Chart(cand_df).mark_point(size=200, shape="diamond").encode(
            x="price:Q", y="positive_ratio:Q", tooltip=["name","price","positive_ratio"]
        )
        st.altair_chart((base + cand_mark).interactive(), use_container_width=True)

        # --- Advanced analytics (sales) — WHAT-IF ---
        with st.expander("📊 Advanced analytics — sales & pricing", expanded=False):
            df_an = add_sales_proxies(wi_df)
            if df_an is None or len(df_an) == 0 or "sales_proxy" not in df_an.columns:
                st.info("Bu panel için 'review_count' kolonu gerekli (sales proxy üretimi). Veri bulunamadı.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Median sales (proxy)", f"{int(np.nanmedian(df_an['sales_proxy'])):,}")
                with c2:
                    med_rev = np.nanmedian(df_an['revenue_proxy']) if 'revenue_proxy' in df_an.columns else np.nan
                    st.metric("Median revenue (proxy)", f"€{med_rev:.0f}" if pd.notna(med_rev) else "—")
                with c3:
                    st.metric("Items with sales", f"{(df_an['sales_proxy']>0).sum():,}")
                with c4:
                    st.metric("Conv. rate (assumed)", "7%")

                reg = sales_regression(df_an)
                st.caption(
                    f"Model: log(sales) ~ log(price) + positive_ratio + year  |  "
                    f"n={reg['n']}  •  R²={reg['r2']:.2f}  •  price elasticity={reg['beta_price']:.2f}  "
                    f"•  ratio effect={reg['beta_ratio']:.2f}"
                )

                plot_df = df_an[["price","sales_proxy","name"]].replace([np.inf,-np.inf], np.nan).dropna()
                plot_df = plot_df[(plot_df["price"]>0) & (plot_df["sales_proxy"]>0)].copy()
                plot_df["log_price"] = np.log(plot_df["price"])
                plot_df["log_sales"] = np.log(plot_df["sales_proxy"])
                if len(plot_df) >= 5:
                    pts = alt.Chart(plot_df).mark_circle(opacity=0.7).encode(
                        x=alt.X("log_price:Q", title="log(Price)"),
                        y=alt.Y("log_sales:Q", title="log(Sales proxy)"),
                        tooltip=["name","price","sales_proxy"]
                    )
                    trend = pts.transform_regression("log_price","log_sales").mark_line()
                    st.altair_chart(pts + trend, use_container_width=True)

                curve = price_response_curve(df_an)
                if len(curve) > 0:
                    st.markdown("**Price → expected sales (holding ratio/year at cohort means)**")
                    ch = alt.Chart(curve).mark_line().encode(
                        x=alt.X("price:Q", title="Price (€)"),
                        y=alt.Y("sales_pred:Q", title="Predicted sales (proxy)")
                    )
                    st.altair_chart(ch, use_container_width=True)

                hm = heat_price_ratio_sales(df_an)
                if len(hm) > 0:
                    heat = alt.Chart(hm).mark_rect().encode(
                        x=alt.X("price_bin:N", title="Price bin"),
                        y=alt.Y("ratio_bin:N", title="Positive ratio bin"),
                        color=alt.Color("median_sales:Q", title="Median sales (proxy)", scale=alt.Scale(scheme="tealblues")),
                        tooltip=["price_bin","ratio_bin","median_sales"]
                    )
                    st.altair_chart(heat, use_container_width=True)

                if "revenue_proxy" in df_an.columns and df_an["revenue_proxy"].notna().sum() > 0:
                    st.markdown("**Revenue proxy histogram**")
                    hist_df = df_an[["revenue_proxy"]].dropna()
                    hist = alt.Chart(hist_df).mark_bar().encode(
                        x=alt.X("revenue_proxy:Q", bin=alt.Bin(maxbins=30), title="Revenue proxy (€)"),
                        y=alt.Y("count()", title="Count")
                    )
                    st.altair_chart(hist, use_container_width=True)

        st.markdown("#### Top matches (first 20)")
        render_cards(wi_df.head(20))
        if st.session_state["show_table"]:
            st.dataframe(wi_df.head(200), use_container_width=True, hide_index=True)

        csv_bytes = wi_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download What-If CSV", data=csv_bytes,
                           file_name="whatif_results.csv", mime="text/csv", type="primary")
# ========= ADVISOR =========
with tab_adv:
    st.subheader("LLM Advisor — Data-aware report")
    st.caption("Bu rapor, What-If’te oluşturduğunuz cohort istatistikleri ve örnek oyunlardan beslenir.")

    wi_df = st.session_state.get("wi_results")
    if wi_df is None or len(wi_df) == 0:
        st.info("Önce **What-If** sekmesinde bir cohort oluşturun.")
    else:
        # SADECE BUTON
        gen = st.button("Generate Advisor Report (AI)", type="primary", use_container_width=True, key="btn_adv_generate")

        if gen:
            try:
                inputs = _advisor.AdvisorInputs(
                    candidate_text="; ".join(st.session_state.get("wi_selected_tags", [])),
                    candidate_price=float(st.session_state.get("wi_price", 14.99)),
                    # scope alanları
                    team_size=int(st.session_state.get("wi_team_size", 3)),
                    dev_months=int(st.session_state.get("wi_dev_months", 12)),
                    risk_tolerance=st.session_state.get("wi_risk", "Medium"),
                    is_2d=(st.session_state.get("wi_graphics") == "2D"),
                    perspective=st.session_state.get("wi_perspective"),
                    online_mode=st.session_state.get("wi_online_mode"),
                    platforms=st.session_state.get("wi_platforms", []),
                    results_df=wi_df,
                )
                report = _advisor.generate_report(inputs, llm_call=groq_client)
                st.session_state["advisor_report"] = report
            except Exception as e:
                st.exception(e)

        # SADECE RAPOR GÖSTERİMİ
        report = st.session_state.get("advisor_report")
        if report:
            rec = report.get("recommendation", {})
            rat = report.get("rationale", {})
            ev  = report.get("evidence", [])

            st.markdown("### Recommendation")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Scope", rec.get("scope_bucket", "—"))
            with c2:
                tl = rec.get("timeline_months") or ["—","—"]
                st.metric("Timeline", f"{tl[0]}–{tl[1]} mo")
            with c3:
                st.metric("List price", f"€{rec.get('list_price_eur', '—')}")

            if rec.get("pricing_notes"):
                st.caption(rec["pricing_notes"])

            st.markdown("### Summary")
            st.write(rat.get("summary", ""))

            if rat.get("tradeoffs"):
                st.markdown("### Trade-offs / Risks")
                for item in rat["tradeoffs"]:
                    st.write(f"- {item}")

            if ev:
                st.markdown("### Comparable Games (Evidence)")
                for e in ev:
                    app_id = e.get("app_id")
                    name = e.get("name", "Game")
                    why  = e.get("why", "")
                    st.write(f"- [{name}](https://store.steampowered.com/app/{app_id}/) — {why}")

            safe_bytes = _json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe).encode("utf-8")
            st.download_button("Download JSON", data=safe_bytes,
                               file_name="advisor_report.json", mime="application/json", type="primary")

# ----------------------------
# Footer
# ----------------------------
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("Modern UI • Embedding + TF-IDF + Hybrid • FAISS FlatIP • Cosine = dot on L2-normed vectors")
