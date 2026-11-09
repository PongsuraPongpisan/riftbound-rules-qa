"""All configuration comes from environment variables. No secrets live in the code.

The OpenAI client itself reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PDF = ROOT / "data" / "raw" / "riftbound_core_rules.pdf"
RULES_JSONL = ROOT / "data" / "processed" / "rules.jsonl"

# The rules version every eval answer was written against (Core Rules, last updated 2026-10-09).
RULES_URL = "https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/5c7946f7c3b8c6e0faff85ab6dadc6d2e721ea94.pdf"
RULES_SHA256 = "94d41fee86f88b61c4157c0389c0595d990b98e5b42555c99d9f649ee64d1fec"
RULES_VERSION = "2026-10-09"

MODEL = os.getenv("LLM_MODEL", "gemma-4-E4B-it")
CHUNKING = os.getenv("CHUNKING", "section")
TOP_K = int(os.getenv("TOP_K", "3"))
# 5.0 = highest floor that loses no answerable question (sweep in results/retrieval_results.md).
MIN_SCORE = float(os.getenv("MIN_SCORE", "5.0"))
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))

REFUSAL = "I couldn't find this in the Core Rules."

# Public deployment: protect the class API key from heavy use.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "6"))   # per visitor IP
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "300"))                      # whole app, per UTC day
