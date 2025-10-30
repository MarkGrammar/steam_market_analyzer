# src/search/brief.py
import re
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

_CAMERA_MAP = {
    "third person": "third-person",
    "third-person": "third-person",
    "3rd person": "third-person",
    "first person": "first-person",
    "first-person": "first-person",
    "1st person": "first-person",
    "top down": "top-down",
    "top-down": "top-down",
    "isometric": "isometric",
    "side": "side",
    "side scroller": "side",
    "sidescroller": "side",
}
_DIM_TOKENS = {"2d": "2d", "3d": "3d"}
_ONLINE_MAP = {
    "mmo": "mmo", "massively multiplayer":"mmo",
    "pvp":"pvp", "versus":"pvp",
    "co-op":"co-op", "coop":"co-op", "co op":"co-op",
    "online":"online"
}
_MUST_TAGS = {
    "metroidvania":["Metroidvania"],
    "roguelike":["Roguelike","Roguelite"],
    "deckbuilder":["Deckbuilding","Card Battler"],
    "soulslike":["Souls-like"],
    "base-building":["Base Building"], "base building":["Base Building"],
    "farming":["Farming Sim"], "cozy":["Cozy"],
    "survival":["Survival"], "crafting":["Crafting"],
    "open world":["Open World"], "pixel":["Pixel Graphics"], "pixel art":["Pixel Graphics"]
}
_STOP_NOT = {"not", "without", "exclude", "except"}

@dataclass
class BriefConstraints:
    camera: Optional[str] = None   # third-person / first-person / top-down / side / isometric
    dim: Optional[str] = None      # 2d / 3d
    online: Optional[str] = None   # none / co-op / pvp / mmo / online
    must: List[str] = None         # must-have tags (Steam tag isimleri)
    must_not: List[str] = None     # must-not kelimeler (negatif sinyaller)
    raw: str = ""

def parse_brief(text: str) -> BriefConstraints:
    t = (text or "").lower()
    cam, dim, online = None, None, None
    for k, v in _CAMERA_MAP.items():
        if k in t:
            cam = v; break
    for k, v in _DIM_TOKENS.items():
        if re.search(rf"\b{k}\b", t):
            dim = v; break
    for k, v in _ONLINE_MAP.items():
        if k in t:
            online = v; break

    must = []
    for k, tags in _MUST_TAGS.items():
        if k in t:
            must += tags
    must = sorted(set(must))

    # kaba "must-not" çıkarımı
    must_not = []
    if "no pvp" in t or "singleplayer only" in t:
        must_not.append("pvp")
    if "no online" in t or "offline only" in t:
        must_not.append("online")

    return BriefConstraints(camera=cam, dim=dim, online=online, must=must, must_not=must_not, raw=text or "")

def hard_filter(df: pd.DataFrame, bc: BriefConstraints) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()

    # Dim / Camera / Online basit eşleşme: tags_str ve genres_str alanlarında arıyoruz (lower)
    def has_token(s: str, token: str) -> bool:
        return token in (s or "").lower()

    if bc.dim == "2d":
        out = out[out["tags_str"].fillna("").str.contains("2d", case=False) |
                  out["genres_str"].fillna("").str.contains("2d", case=False)]
    elif bc.dim == "3d":
        out = out[out["tags_str"].fillna("").str.contains("3d", case=False) |
                  out["genres_str"].fillna("").str.contains("3d", case=False)]

    if bc.camera:
        token = bc.camera.replace("-", " ")
        mask = (out["tags_str"].fillna("").str.contains(token, case=False)) | \
               (out["genres_str"].fillna("").str.contains(token, case=False))
        # side için ekstra anahtar kelimeler:
        if bc.camera == "side":
            mask = mask | out["tags_str"].fillna("").str.contains("side-scroller|sidescroller|platformer", case=False, regex=True)
        out = out[mask]

    if bc.online:
        tok = bc.online
        patt = {"co-op": "co[- ]?op", "pvp": "pvp", "mmo": "mmo|massively", "online":"online"}.get(tok, tok)
        mask = out["tags_str"].fillna("").str.contains(patt, case=False, regex=True) | \
               out["genres_str"].fillna("").str.contains(patt, case=False, regex=True)
        out = out[mask]

    # must-have Steam tag’leri
    for tag in (bc.must or []):
        out = out[out["tags_str"].fillna("").str.contains(re.escape(tag), case=False)]

    # must-not (kaba)
    for ban in (bc.must_not or []):
        out = out[~(out["tags_str"].fillna("").str.contains(ban, case=False))]

    return out
