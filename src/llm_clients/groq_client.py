from __future__ import annotations

import os, json, requests
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Proje kökünden .env'yi garanti yükle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _get_env(key: str, default: Optional[str] = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val

def groq_client(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 800,
) -> str:
    api_key = _get_env("GROQ_API_KEY")
    model   = _get_env("GROQ_MODEL", "llama-3.1-8b-instant")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected Groq response: {data}") from e
