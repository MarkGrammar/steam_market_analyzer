# src/analytics/stats.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- Güvenli yardımcılar ----------
def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or len(df) == 0 or col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def iqr(x: pd.Series) -> float:
    if len(x) == 0: return np.nan
    q1, q3 = np.nanpercentile(x, [25, 75])
    return float(q3 - q1)

def mad(x: pd.Series) -> float:
    if len(x) == 0: return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))  # “median absolute deviation”

def quantiles(x: pd.Series, qs=(5, 25, 50, 75, 95)) -> dict:
    if len(x) == 0: return {f"q{q}": np.nan for q in qs}
    vals = np.nanpercentile(x, qs)
    return {f"q{q}": float(v) for q, v in zip(qs, vals)}

# ---------- Ana özet ----------
def cohort_summary(df: pd.DataFrame) -> Dict[str, Any]:
    # Guard against None or empty DataFrame
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return dict(
            n_items=0,
            price_median=np.nan,
            price_mad=np.nan,
            ratio_median=np.nan,
            year_median=np.nan,
        )

    # Safe selectors
    p = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce")
    r = pd.to_numeric(df.get("positive_ratio", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(df.get("release_year", pd.Series(dtype=float)), errors="coerce")

    # Median Absolute Deviation (robust spread) for price
    pm = p.median(skipna=True)
    mad = (p - pm).abs().median(skipna=True) if pd.notna(pm) else np.nan

    return dict(
        n_items=int(len(df)),
        price_median=float(pm) if pd.notna(pm) else np.nan,
        price_mad=float(mad) if pd.notna(mad) else np.nan,
        ratio_median=float(r.median(skipna=True)) if len(r.dropna()) else np.nan,
        year_median=int(y.median(skipna=True)) if len(y.dropna()) else np.nan,
    )

# ---------- Fiyat bantları ----------
def price_bands(px: pd.Series, step=2.0, max_price=60.0) -> pd.DataFrame:
    """0–max aralığını step adımlarla kovala; histogram için hazır döner."""
    if px is None:
        return pd.DataFrame(columns=["band", "count"])

    s = pd.to_numeric(px, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["band", "count"])

    # 0..max aralığına kırp
    s = s.clip(lower=0, upper=float(max_price))

    # Kenarları üret (ör. 0,2,4,...,60)
    edges = np.arange(0.0, float(max_price) + step, step)
    if edges.size < 2:  # step çok büyükse güvenlik
        edges = np.array([0.0, float(max_price)])

    # Binleme: sol kapalı, sağ açık => [0,2), [2,4), ...
    cats = pd.cut(s, bins=edges, right=False, include_lowest=True)

    # Sayımlar (sıra bozulmadan)
    vc = cats.value_counts(sort=False)

    # Etiketleri güvenli üret
    bands = [f"€{iv.left:.0f}–€{iv.right:.0f}" for iv in vc.index]

    return pd.DataFrame({"band": bands, "count": vc.values})

# ---------- Bootstrap CI ----------
def bootstrap_ci_mean(x: pd.Series, iters=2000, alpha=0.05, seed=42) -> tuple[float,float]:
    """Ortalamaya %95 güven aralığı (basic bootstrap)."""
    x = x.dropna().values
    if x.size < 8:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    n = x.size
    for _ in range(iters):
        samp = rng.choice(x, size=n, replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return float(lo), float(hi)

# ---------- Basit “regression-lite” ----------
def ratio_vs_price_year(df: pd.DataFrame) -> dict:
    """
    positive_ratio ~ price + release_year için lineer model (saf NumPy).
    Sadece işaret/yön görmek için hafif özet döner.
    """
    y = _safe_series(df, "positive_ratio")
    p = _safe_series(df, "price")
    t = _safe_series(df, "release_year")
    mask = (~y.isna()) & (~p.isna()) & (~t.isna())
    y, p, t = y[mask].values, p[mask].values, t[mask].values
    if y.size < 20:
        return dict(n=int(y.size), beta_price=np.nan, beta_year=np.nan, r2=np.nan)
    X = np.c_[np.ones_like(p), p, t]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return dict(n=int(y.size), beta_price=float(beta[1]), beta_year=float(beta[2]), r2=float(r2))


# ---------- Satış & gelir proxy'leri ----------
def add_sales_proxies(
    df: pd.DataFrame,
    review_col: str = "review_count",
    conv_rate: float = 0.07,  # reviews/sales dönüşüm katsayısı
    price_col: str = "price",
) -> pd.DataFrame:
    """reviews → sales ve revenue proxy kolonlarını ekler. Kolon yoksa dokunmaz."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if review_col in out.columns:
        rv = pd.to_numeric(out[review_col], errors="coerce")
        sales = rv.fillna(0).clip(lower=0) / float(conv_rate)
        out["sales_proxy"] = sales
        if price_col in out.columns:
            out["revenue_proxy"] = pd.to_numeric(out[price_col], errors="coerce").fillna(0) * sales
    return out

# ---------- Yardımcı: güvenli kolon çekici ----------
def _s(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or len(df) == 0 or col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

# ---------- Satış regresyonları ----------
def sales_regression(df: pd.DataFrame) -> dict:
    """
    log(sales) ~ log(price) + positive_ratio + release_year
    Dönenler: n, beta_price (elastisite), beta_ratio, beta_year, r2
    """
    if df is None or len(df) == 0 or "sales_proxy" not in df.columns:
        return dict(n=0, beta_price=np.nan, beta_ratio=np.nan, beta_year=np.nan, r2=np.nan)

    s = _s(df, "sales_proxy")
    p = _s(df, "price")
    r = _s(df, "positive_ratio")
    y = _s(df, "release_year")

    mask = (~s.isna()) & (~p.isna()) & (p > 0) & (~r.isna()) & (~y.isna())
    if mask.sum() < 25:
        return dict(n=int(mask.sum()), beta_price=np.nan, beta_ratio=np.nan, beta_year=np.nan, r2=np.nan)

    # Tasarım matrisi (log-log fiyat; diğerleri lineer)
    log_s = np.log(s[mask].values)
    log_p = np.log(p[mask].values)
    R = r[mask].values
    Y = y[mask].values
    X = np.c_[np.ones_like(log_p), log_p, R, Y]

    beta, *_ = np.linalg.lstsq(X, log_s, rcond=None)
    pred = X @ beta
    ss_res = np.sum((log_s - pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(
        n=int(mask.sum()),
        beta_price=float(beta[1]),   # fiyat elastisitesi
        beta_ratio=float(beta[2]),   # ratio etkisi (%1 artış → log-sales etkisi)
        beta_year=float(beta[3]),
        r2=float(r2),
    )

def price_response_curve(df: pd.DataFrame, grid: int = 30) -> pd.DataFrame:
    """
    Yukarıdaki modelle “fiyat → beklenen satış” eğrisi (cohort ortalama ratio & yıl ile).
    """
    meta = sales_regression(df)
    if not np.isfinite(meta.get("beta_price", np.nan)):
        return pd.DataFrame(columns=["price","sales_pred"])

    # Grid: mevcut fiyatların p5–p95 arası
    p = _s(df, "price")
    p = p[(~p.isna()) & (p > 0)]
    if len(p) < 10:
        return pd.DataFrame(columns=["price","sales_pred"])
    lo, hi = np.nanpercentile(p, [5, 95])
    prices = np.linspace(lo, hi, grid)

    # Ortalama kontrol değerleri
    R = np.nanmean(_s(df, "positive_ratio"))
    Y = np.nanmean(_s(df, "release_year"))
    # Ortalama log-sales düzeyi için sabiti yakalayalım
    # Basit yaklaşım: Ortalama log(sales) - b*ortalama X
    mask = (~_s(df, "sales_proxy").isna()) & (~p.isna()) & (p > 0)
    log_s = np.log(_s(df, "sales_proxy")[mask].values)
    log_p = np.log(_s(df, "price")[mask].values)
    Rv = _s(df, "positive_ratio")[mask].values
    Yv = _s(df, "release_year")[mask].values
    # Aynı beta'ları kullanıp sabiti geri çözelim:
    b_price, b_ratio, b_year = meta["beta_price"], meta["beta_ratio"], meta["beta_year"]
    const = np.mean(log_s - (b_price*log_p + b_ratio*Rv + b_year*Yv))

    sales_pred = np.exp(const + b_price*np.log(prices) + b_ratio*R + b_year*Y)
    return pd.DataFrame({"price": prices, "sales_pred": sales_pred})

# ---------- Isı haritası için kova istatistik ----------
def heat_price_ratio_sales(df: pd.DataFrame, bins_price=12, bins_ratio=10) -> pd.DataFrame:
    """
    price × positive_ratio ızgarasında median sales_proxy ısı haritası verisi.
    """
    if df is None or len(df) == 0 or "sales_proxy" not in df.columns:
        return pd.DataFrame(columns=["price_bin","ratio_bin","median_sales"])
    P = _s(df, "price")
    R = _s(df, "positive_ratio")
    S = _s(df, "sales_proxy")
    mask = (~P.isna()) & (~R.isna()) & (~S.isna()) & (P >= 0)
    P, R, S = P[mask], R[mask], S[mask]

    if len(P) < 20:
        return pd.DataFrame(columns=["price_bin","ratio_bin","median_sales"])

    pb = pd.qcut(P, q=bins_price, duplicates="drop")
    rb = pd.qcut(R, q=bins_ratio, duplicates="drop")
    g = pd.DataFrame({"pb": pb, "rb": rb, "S": S}).groupby(["pb","rb"])["S"].median().reset_index()
    # Bin label'larını stringle
    def _lab(cat):
        try:
            lo, hi = cat.left, cat.right
            return f"{lo:.2g}–{hi:.2g}"
        except Exception:
            return str(cat)
    g["price_bin"] = g["pb"].apply(_lab)
    g["ratio_bin"] = g["rb"].apply(_lab)
    return g.rename(columns={"S":"median_sales"})[["price_bin","ratio_bin","median_sales"]]
