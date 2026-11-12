"""Model tiers and retired/deprecated model ids, from the dated data file data/models.json.
No prices and no network: the file is maintained by hand and carries its own as-of date."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "models.json"
PREFIXES = ("openai/", "anthropic/", "gemini/", "google/", "vertex_ai/", "models/", "azure/", "openrouter/openai/",
            "openrouter/anthropic/", "openrouter/google/")


@lru_cache(maxsize=1)
def data() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def as_of() -> str:
    return data()["as_of"]


def normalize(model: str) -> str:
    m = model.strip()
    # LangChain init_chat_model style "openai:gpt-4o"
    if re.match(r"^(openai|anthropic|google_genai|google_vertexai|azure_openai):", m):
        m = m.split(":", 1)[1]
    for p in PREFIXES:
        if m.startswith(p):
            m = m[len(p):]
            break
    return m


@lru_cache(maxsize=None)
def _retired_map() -> dict:
    return {r["id"]: r for r in data()["retired"]}


def retired(model: str) -> dict | None:
    return _retired_map().get(normalize(model))


def top_tier(model: str) -> dict | None:
    m = normalize(model)
    for t in data()["top_tier"]:
        if re.search(t["pattern"], m):
            return t
    return None


def rejects_sampling(model: str) -> bool:
    m = normalize(model)
    return any(re.search(p, m) for p in data()["no_sampling_params"]["patterns"])
