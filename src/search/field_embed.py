# src/search/field_embed.py
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    faiss.normalize_L2(x)
    return x

class FieldEmbedder:
    def __init__(self, model_name: str, w_name=0.6, w_tags=1.3, w_genres=1.0, w_desc=0.4):
        self.model = SentenceTransformer(model_name)
        self.w = dict(name=w_name, tags=w_tags, genres=w_genres, desc=w_desc)

    def encode_row(self, name: str, tags: str, genres: str, desc: str) -> np.ndarray:
        parts = {
            "name":   (name or ""),
            "tags":   (tags or ""),
            "genres": (genres or ""),
            "desc":   (desc or "")
        }
        vecs, ws = [], []
        for k, txt in parts.items():
            if not txt.strip():
                continue
            v = self.model.encode([txt], convert_to_numpy=True, normalize_embeddings=False)[0]
            vecs.append(v); ws.append(self.w.get(k, 1.0))
        if not vecs:
            return np.zeros((self.model.get_sentence_embedding_dimension(),), dtype="float32")
        V = np.vstack(vecs)
        w = np.array(ws, dtype="float32")[:, None]
        out = (V * w).sum(axis=0) / (w.sum(axis=0)+1e-8)
        return _norm(out[None, :])[0]


# src/search/field_embed.py
from typing import Dict, List, Optional

# Basit ağırlıklandırma: embedding tarafında grafik/perspective sinyalini öne çıkaralım
def build_query_text_fields(
    tags: List[str],
    perspective: Optional[str] = None,
    graphics: Optional[str] = None,
) -> Dict[str, str]:
    tags_txt = " ".join(tags or [])
    aux_bits = []
    if graphics:
        aux_bits.append(graphics)
    if perspective:
        aux_bits.append(perspective)

    # Embedding kanalı için biraz ağırlık
    dense_text = " ".join([tags_txt] + aux_bits + aux_bits)  # aux'u iki kez ekle -> sinyali büyüt
    tfidf_text = " ".join([tags_txt] + aux_bits)

    return {
        "dense_text": dense_text.strip(),
        "tfidf_text": tfidf_text.strip(),
        "raw": {
            "tags": tags or [],
            "graphics": graphics,
            "perspective": perspective,
        }
    }

# Geriye dönük uyumluluk: yanlış importları kırmamak için alias
build_query_text_fieldsw = build_query_text_fields
