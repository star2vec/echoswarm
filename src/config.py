"""
src/config.py — Hermes Engine provider configuration.

To swap from Groq (dev) to Anthropic (production), change LLM_PROVIDER here,
or set the LLM_PROVIDER environment variable. Everything else is automatic.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── Provider Selection ────────────────────────────────────────────────────────
# "groq"       → Groq API  (llama-3.1-70b-versatile) — default for dev/simulation
# "anthropic"  → Anthropic API (claude-sonnet-4-6)   — production swap
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

# ── Groq (default: development + simulation) ──────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"  # clarity validator

# ── Anthropic (production swap) ───────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MAIN_MODEL: str = "claude-3-5-sonnet-20240620"  # or claude-sonnet-4-6
ANTHROPIC_FAST_MODEL: str = "claude-haiku-4-5-20251001"   # clarity validator

# ── Hermes Engine Limits ──────────────────────────────────────────────────────
HERMES_MAX_RETRIES: int = 3
HERMES_CLARITY_PASS_THRESHOLD: int = 7
HERMES_MAX_TOKENS: int = 1024    # main generation budget
HERMES_FAST_MAX_TOKENS: int = 256  # clarity validator budget
