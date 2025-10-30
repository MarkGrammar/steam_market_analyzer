# src/features/advisor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

# ---------------------------------------
# Inputs (What-If + cohort)
# ---------------------------------------
@dataclass
class AdvisorInputs:
    candidate_text: str
    candidate_price: float
    candidate_year: Optional[int] = None

    # What-If / Scope alanları (NEW)
    team_size: int = 3
    dev_months: int = 12
    risk_tolerance: str = "Medium"           # Low / Medium / High
    is_2d: Optional[bool] = None             # True=2D, False=3D, None=unknown
    perspective: Optional[str] = None        # Side Scroller, Top-Down, ...
    online_mode: Optional[str] = None        # None, Co-op, PvP, MMO
    platforms: Optional[List[str]] = None    # ["PC","Console",...]

    # Eski mini-brief alanları (geriye dönük)
    singleplayer: Optional[bool] = None
    session_length: Optional[str] = None
    team_target: Optional[str] = None

    # Zorunlu: filtrelenmiş kohort
    results_df: Optional[pd.DataFrame] = None


# ---------------------------------------
# Küçük yardımcılar
# ---------------------------------------
def _cohort_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Kohortun temel istatistiklerini çıkar (median, Q1, Q3)."""
    if df is None or len(df) == 0:
        return dict(
            median_price=np.nan,
            q1_price=np.nan,
            q3_price=np.nan,
            median_ratio=np.nan,
            median_year=np.nan,
        )

    s_price = df["price"].dropna()
    s_ratio = df["positive_ratio"].dropna()
    s_year  = df["release_year"].dropna()

    def q(s, p):
        return float(np.nanquantile(s, p)) if len(s) else np.nan

    return dict(
        median_price=float(np.nanmedian(s_price)) if len(s_price) else np.nan,
        q1_price=q(s_price, 0.25) if len(s_price) else np.nan,
        q3_price=q(s_price, 0.75) if len(s_price) else np.nan,
        median_ratio=float(np.nanmedian(s_ratio)) if len(s_ratio) else np.nan,
        median_year=int(np.nanmedian(s_year)) if len(s_year) else np.nan,
    )


def _pick_examples(df: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
    """LLM'e kanıt olarak verilecek 2–3 oyun seç (benzerlik / sinyal)."""
    if df is None or len(df) == 0:
        return []

    if "similarity" in df.columns:
        tmp = df.sort_values("similarity", ascending=False)
    else:
        tmp = df.copy()
        if "positive_ratio" in tmp.columns:
            tmp = tmp.sort_values("positive_ratio", ascending=False)

    tmp = tmp.head(n).copy()
    out = []
    for _, r in tmp.iterrows():
        why_bits = []
        if pd.notna(r.get("price")):
            why_bits.append(f"price €{float(r['price']):.2f}")
        if pd.notna(r.get("positive_ratio")):
            why_bits.append(f"ratio {float(r['positive_ratio'])*100:.1f}%")
        if r.get("tags_str"):
            why_bits.append(f"tags: {', '.join(r['tags_str'].split(', ')[:2])}")

        out.append(dict(
            app_id=int(r["app_id"]),
            name=str(r.get("name", "Game")),
            why="; ".join(why_bits) if why_bits else "relevant example in the cohort",
        ))
    return out


# ---------------------------------------
# LLM Promptları
# ---------------------------------------
_SYSTEM_PROMPT = (
    "You are a senior game publishing analyst. "
    "Use data-driven reasoning with the provided cohort statistics and comparables. "
    "Be specific, avoid filler, and make actionable recommendations."
)

def _compose_user_prompt(
    inputs: AdvisorInputs,
    cohort_stats: Dict[str, Any],
    examples: List[Dict[str, Any]],
) -> str:
    examples_block = "\n".join(
        f"- {e['name']} (https://store.steampowered.com/app/{e['app_id']}/): {e['why']}"
        for e in examples
    ) or "- (no comparable examples available)"

    med_price = cohort_stats["median_price"]
    q1 = cohort_stats["q1_price"]
    q3 = cohort_stats["q3_price"]
    med_ratio = cohort_stats["median_ratio"]
    med_year = cohort_stats["median_year"]

    plat = ", ".join(inputs.platforms or [])

    graphics = "2D" if inputs.is_2d is True else ("3D" if inputs.is_2d is False else "N/A")
    perspective = inputs.perspective or "N/A"
    online = inputs.online_mode or "N/A"
    # >>> YENİ: yıl satırlarını sadece varsa ekle
    year_line = f"- Release year: {inputs.candidate_year}\n" if inputs.candidate_year is not None else ""
    cohort_year_line = f"- Release year median {int(med_year)}\n" if (isinstance(med_year, (int, float)) and med_year == med_year) else ""

    return f"""
CANDIDATE
- Concept tags: {inputs.candidate_text}
- Price: €{inputs.candidate_price:.2f}
- Release year: {inputs.candidate_year}

SCOPE
- Team size: {inputs.team_size}
- Duration (months): {inputs.dev_months}
- Risk tolerance: {inputs.risk_tolerance}
- Graphics: {graphics}
- Perspective: {perspective}
- Online: {online}
- Platforms: {plat or "PC (assumed)"}

COHORT STATS
- Price median €{med_price:.2f} (Q1–Q3: €{q1:.2f}–€{q3:.2f})
- Positive ratio median {med_ratio*100:.1f}%
- Release year median {med_year}

COMPARABLES
{examples_block}

Write a structured report:
1) Executive Summary (3–5 sentences).
2) Market Positioning (relative to cohort price/rating patterns).
3) Scope & Timeline sanity check given team/duration/online/graphics (flag concrete risks).
4) Pricing recommendation: a single EUR number with rationale vs cohort stats.
5) Trade-offs / Risks: 3–5 bullets tied to data.
6) Evidence: 3 bullets (name + why).
Keep it concise and professional.
""".strip()


# ---------------------------------------
# Basit heuristikler (UI metrikleri)
# ---------------------------------------
def _heuristic_scope_and_timeline(inputs: AdvisorInputs) -> tuple[str, list[int]]:
    """Team/online/2D-3D ve süreye göre kaba scope+timeline tahmini."""
    # Başlangıç varsayımı
    scope = "Small"
    tmin, tmax = 12, 18

    # 3D + online projeler daha maliyetli/süreli
    if inputs.is_2d is False:
        tmin += 3; tmax += 3
    if (inputs.online_mode or "None") not in ("None", ""):
        tmin += 3; tmax += 6

    # ekip büyüklüğüne göre ayar
    if inputs.team_size <= 2:
        scope = "Small"
        tmin += 3; tmax += 3
    elif inputs.team_size <= 5:
        scope = "Small"
    elif inputs.team_size <= 10:
        scope = "Mid"
    else:
        scope = "Mid+"

    # Kullanıcı planı (dev_months) belirginse bandı ona yakınla
    if inputs.dev_months:
        # planlanan süre etrafında ±3 ay esneklik
        tmin = max(6, int(inputs.dev_months) - 3)
        tmax = int(inputs.dev_months) + 3

    return scope, [tmin, tmax]


def _heuristic_price(inputs: AdvisorInputs, stats: Dict[str, Any]) -> tuple[float, str]:
    """Median’a göre fiyatı makul banda çeker ve not döner."""
    target = float(inputs.candidate_price)
    if not np.isnan(stats["median_price"]):
        med = stats["median_price"]
        if target > med * 1.25:
            target = round(med * 1.10, 2)
            note = "Adjusted down towards cohort median."
        elif target < med * 0.75:
            target = round(med * 0.90, 2)
            note = "Adjusted up towards cohort median."
        else:
            note = "Price sits near cohort norms."
    else:
        note = "No cohort price stats; kept candidate price."
    return target, note


# ---------------------------------------
# Ana API: generate_report
# ---------------------------------------
def generate_report(inputs: AdvisorInputs, llm_call=None) -> Dict[str, Any]:
    """
    Raporu üretir.
    - llm_call(system, user, ...) verilirse Groq (veya uyumlu) modelden doğal dil rapor alınır.
    - UI metrikleri için basit ama makul heuristikler uygulanır.
    Dönen yapı Streamlit UI'nin beklediği sözleşmeye uygundur.
    """
    df = inputs.results_df if inputs.results_df is not None else pd.DataFrame()
    stats = _cohort_stats(df)
    examples = _pick_examples(df, n=3)

    # Heuristik metrikler
    scope_bucket, timeline_months = _heuristic_scope_and_timeline(inputs)
    target_price, pricing_note = _heuristic_price(inputs, stats)

    # LLM raporu (varsa)
    if llm_call is not None:
        user_prompt = _compose_user_prompt(inputs, stats, examples)
        try:
            # DİKKAT: llm_call(system_prompt, user_prompt, **kwargs) şeklinde çağırıyoruz
            summary_text = llm_call(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
                top_p=0.9,
                max_tokens=900,
)
        except Exception:
            summary_text = (
                "LLM call failed; fallback summary: Keep scope tight, "
                "align price to cohort medians, and de-risk online/3D features."
            )
    else:
        summary_text = (
            "Rule-based summary: Small-scope indie positioning. "
            "Price near cohort medians; focus on polish and schedule control."
        )

    # Evidence
    evidence = examples

    # Trade-offs (kısa maddeler)
    tradeoffs = [
        "Online features and 3D art increase schedule & tooling risk.",
        "Small team requires narrow feature scope and strong core loop.",
    ]

    return {
        "recommendation": {
            "scope_bucket": scope_bucket,
            "timeline_months": timeline_months,
            "list_price_eur": f"{target_price:.2f}",
            "pricing_notes": pricing_note,
        },
        "rationale": {
            "summary": summary_text,
            "tradeoffs": tradeoffs,
            "method": "cohort_stats + comparables + LLM" if llm_call else "cohort_stats + comparables (rule-based)",
        },
        "evidence": evidence,
        "cohort_stats": stats,
    }
