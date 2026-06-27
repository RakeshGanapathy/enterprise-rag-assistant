"""
Enterprise query router: classifies every incoming question and selects
the optimal retrieval strategy before any vector or keyword search runs.

Design principles:
1. Normalize first  — strip, deduplicate whitespace, truncate, reject garbage
2. Classify second  — deterministic regex signals, no LLM call
3. Never throw      — any exception returns the safe fallback (hybrid)
4. Always log why   — reason string goes into workflow_steps for observability

Routing signals (deterministic, zero latency):
  exact_match  -> quoted phrases, policy IDs (HR-001), alphanumeric codes  -> hybrid
  keyword_rich -> short dense queries, acronyms, product names             -> hybrid
  conceptual   -> long natural-language questions                           -> semantic
  default      -> hybrid  (enterprise safe default — never loses a keyword match)
"""
from __future__ import annotations

import re
import unicodedata

# ── normalization constants ──────────────────────────────────────────────────
_MAX_QUERY_CHARS = 1000          # hard cap before any processing
_MIN_ALPHA_RATIO = 0.30          # at least 30% of chars must be letters/digits
_MIN_MEANINGFUL_WORDS = 1        # reject pure punctuation / whitespace queries

# ── classification signals ───────────────────────────────────────────────────
_QUOTED_PHRASE      = re.compile(r'"[^"]+"')
_POLICY_ID          = re.compile(r"\b[A-Za-z]+-\d+\b")          # HR-001, SEC-2024
_ALPHANUMERIC_CODE  = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")      # SOC2, SAML, ISO, API
_VERSION_TAG        = re.compile(r"\bv?\d+\.\d+\b")             # v2.0, 3.1

# word-count thresholds
_KEYWORD_RICH_MAX_WORDS = 6
_CONCEPTUAL_MIN_WORDS   = 15

# safe fallback used for any error or ambiguous case
_SAFE_MODE   = "hybrid"
_SAFE_REASON = "router:safe_fallback->hybrid"


# ── public API ───────────────────────────────────────────────────────────────

def classify_query(question: str) -> tuple[str, str]:
    """
    Classify a query and return (search_mode, reason).

    search_mode : "hybrid" | "semantic"
    reason      : human-readable, logged into workflow_steps

    Never raises. Any unexpected input or internal error returns hybrid.
    """
    try:
        normalized, rejection_reason = _normalize(question)
        if rejection_reason:
            # Input was garbage — still serve the request, just log it
            return _SAFE_MODE, f"router:rejected_input({rejection_reason})->hybrid"
        return _classify(normalized)
    except Exception as exc:  # noqa: BLE001
        return _SAFE_MODE, f"router:exception({type(exc).__name__})->hybrid"


# ── normalization ─────────────────────────────────────────────────────────────

def _normalize(raw: str) -> tuple[str, str]:
    """
    Clean and validate raw query text.
    Returns (normalized_text, rejection_reason).
    rejection_reason is empty string when input is valid.
    """
    if not isinstance(raw, str):
        return "", "not_a_string"

    # Hard length cap — prevents regex catastrophic backtracking and BM25 memory spike
    text = raw[:_MAX_QUERY_CHARS]

    # Normalize unicode to composed form, strip control characters
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C") or ch in " \t")

    # Collapse whitespace
    text = " ".join(text.split())

    if not text:
        return "", "empty_after_normalization"

    # Reject if too few real word characters (e.g. "??????????", "!!!!!")
    alpha_count = sum(1 for ch in text if ch.isalnum())
    if len(text) > 0 and alpha_count / len(text) < _MIN_ALPHA_RATIO:
        return "", f"low_alpha_ratio({alpha_count}/{len(text)})"

    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if len(words) < _MIN_MEANINGFUL_WORDS:
        return "", "no_meaningful_words"

    return text, ""


# ── classification ────────────────────────────────────────────────────────────

def _classify(text: str) -> tuple[str, str]:
    """Apply classification rules to a normalized, validated query."""
    # Check patterns against original casing (codes/IDs are case-sensitive signals)
    if _QUOTED_PHRASE.search(text):
        return "hybrid", "router:exact_phrase->hybrid"

    if _POLICY_ID.search(text):
        return "hybrid", "router:policy_id->hybrid"

    if _ALPHANUMERIC_CODE.search(text):
        return "hybrid", "router:acronym_or_code->hybrid"

    if _VERSION_TAG.search(text):
        return "hybrid", "router:version_tag->hybrid"

    word_count = len(text.split())

    if word_count <= _KEYWORD_RICH_MAX_WORDS:
        return "hybrid", f"router:short_query({word_count}w)->hybrid"

    if word_count >= _CONCEPTUAL_MIN_WORDS:
        return "semantic", f"router:long_conceptual({word_count}w)->semantic"

    # Mid-length with no strong signal — hybrid is enterprise safe default
    return "hybrid", f"router:default({word_count}w)->hybrid"
