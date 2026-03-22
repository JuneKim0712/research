#!/usr/bin/env python3
"""
Step 2: Build cue-based candidate windows for explicit competition evidence extraction.

Reads a manifest (JSON or CSV) and corresponding extracted section text files only.
Splits text into sentences, detects competition-related cue phrases using a
four-group hierarchy (strict_explicit, contextual_explicit, implicit_or_broad,
demotion_or_negative), generates short candidate windows, scores/prioritizes them,
applies overlap-aware deduplication, and exports structured output for downstream
explicit company extraction. Retains metadata for future implicit/profile stages.

No NER, no alias resolution, no CIK linking beyond manifest, no competitor
prediction, no embeddings, no ML — heuristic text windowing only.

Outputs:
  candidate_windows.csv
  candidate_windows.jsonl
  window_audit_summary.txt (audit content only; no stray print output)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Cue group names (used in output and audit)
CUE_GROUP_STRICT = "strict_explicit"
CUE_GROUP_CONTEXTUAL = "contextual_explicit"
CUE_GROUP_IMPLICIT = "implicit_or_broad"
CUE_GROUP_DEMOTION = "demotion_or_negative"
CUE_GROUP_HEADING_FALLBACK = "heading_fallback_broad"

# Export bucket names (separate output pools)
EXPORT_BUCKET_STRICT = "strict_explicit"
EXPORT_BUCKET_CONTEXTUAL = "contextual_explicit"
EXPORT_BUCKET_BROAD = "broad_or_implicit"

# =============================================================================
# Four-group cue system
# =============================================================================
#
# strict_explicit_cues: Strongest signal for direct competitor lists / named rivals.
#   Priority 4. Window: trigger sentence only, or prev+current+next only if under
#   competition heading.
#
# contextual_explicit_cues: Strong competition language; upgraded when under
#   competition heading or when a strict-explicit cue appears nearby.
#   Priority 3 when upgraded, else 2. Window: prev+current+next.
#
# implicit_or_broad_cues: Market/alternative/strategy language; kept for future
#   implicit track. Priority 1. Window: trigger sentence only.
#
# demotion_or_negative_context_cues: Product catalog, customer/partner examples,
#   regulation, generic "competitive advantage" language. Demote or suppress.
#

STRICT_EXPLICIT_PATTERNS: list[str] = [
    r"competitors\s+include",
    r"principal\s+competitors",
    r"primary\s+competitors",
    r"we\s+compete\s+with",
    r"we\s+face\s+competition\s+from",
    r"face\s+competition\s+from",
    r"companies\s+that\s+have\s+introduced\s+(?:products?|services?|technologies?)\s+include",
    r"the\s+following\s+companies\s+(?:are|include|compete)",
    r"our\s+(?:principal|primary|major|key|direct)\s+competitors",
    r"competition\s+from\s+(?:companies\s+such\s+as|companies\s+including)",
    r"major\s+competitors\s+include",
    r"key\s+competitors\s+include",
    r"direct\s+competitors\s+include",
    r"competition\s+comes\s+from",
    r"compete\s+against",
    r"competing\s+with",
    r"compete\s+directly\s+with",
    r"significant\s+competitors",
    r"competitive\s+(?:threat|threats)\s+from",
    r"compete\s+in\s+(?:the\s+)?(?:same|these|our)\s+markets?\s+with",
]

CONTEXTUAL_EXPLICIT_PATTERNS: list[str] = [
    r"competition\s+from",
    r"main\s+competitiors",
    r"our\s+competitors",
    r"competition\s+in\s+the",
    r"the\s+competition\s+in\s+the",
    r"highly\s+competitive",
    r"intensely\s+competitive",
    r"intense\s+competition",
    r"pricing\s+pressure",
    r"price\s+pressure",
    r"market\s+share",
    r"\bcompetitors?\b",
    r"\bcompete\b",
    r"\brivals?\b",
    r"competitive\s+(?:environment|landscape|conditions|pressures?|dynamics)",
    r"competition\s+in\s+(?:our|the)",
    r"compete\s+on\s+(?:the\s+basis|price|quality|performance|features?)",
]

IMPLICIT_OR_BROAD_PATTERNS: list[str] = [
    r"installed\s+base",
    r"technical\s+resources?",
    r"marketing\s+resources?",
    r"distribution\s+(?:channel|network|capabilities?)",
    r"switching\s+costs?",
    r"barriers?\s+to\s+entry",
    r"wallet\s+share",
    r"competitive\s+advantage",
    r"competitive\s+position",
    r"ability\s+to\s+compete",
    r"effectively\s+compete",
]

DEMOTION_PATTERNS: list[str] = [
    r"customers?\s+include",
    r"clients?\s+include",
    r"examples?\s+include",
    r"such\s+as\b",
    r"for\s+example",
    r"services?\s+(?:such\s+as|including|like)",
    r"offered\s+by",
    r"used\s+by",
    r"through\s+(?:brands?|companies|firms)\s+including",
    r"third[- ]party\s+(?:services?|providers?|platforms?)",
    r"partners?\s+include",
    r"suppliers?\s+include",
    r"vendors?\s+include",
    r"regulatory\s+(?:requirements?|environment|approval)",
    r"patent(s)?\s+(?:protection|portfolio|litigation)",
    r"compliance\s+with",
    r"competitive\s+advantage\s+(?:that|we|our)",
    r"our\s+competitive\s+position",
    r"product\s+(?:lines?|portfolio|offerings?)\s+(?:include|such as)",
    r"platform(s)?\s+(?:include|such as|offered by)",
]

# Priority scores
PRIORITY_STRICT = 4
PRIORITY_CONTEXTUAL_UPGRADED = 3
PRIORITY_CONTEXTUAL = 2
PRIORITY_IMPLICIT = 1
PRIORITY_DEMOTED = 0
PRIORITY_COMPETITION_HEADING_BONUS = 1
PRIORITY_HEADING_FALLBACK = 0

# Window expansion (prev_sentences, next_sentences)
WINDOW_STRICT = (0, 0)  # trigger only by default
WINDOW_STRICT_UNDER_HEADING = (1, 1)
WINDOW_CONTEXTUAL = (1, 1)
WINDOW_IMPLICIT = (0, 0)
WINDOW_HEADING_EXTRA = 1
WINDOW_MAX_SENTENCES = 5

# Overlap threshold: if two windows share this fraction of (char) content, keep one
OVERLAP_THRESHOLD = 0.75

# Headings
COMPETITION_HEADING_PATTERNS: list[str] = [
    r"(?mi)^\s*(item\s*1\s*[.\-\u2013\u2014:]?\s*)?(our\s+)?competition\s*$",
    r"(?mi)^\s*competitive\s+(?:environment|landscape|conditions|overview|dynamics)\s*$",
    r"(?mi)^\s*competitive\s+conditions\s*$",
    r"(?mi)^\s*market\s+competition\s*$",
]

# Broad but still competition-specific heading detector used only for fallback windows.
# This does not alter strict/contextual cue matching rules.
HEADING_FALLBACK_PATTERNS: list[str] = [
    r"(?i)\bcompetition\b",
    r"(?i)\bcompetitive\b",
    r"(?i)\bcompetitive\s+landscape\b",
    r"(?i)\bmarket\s+competition\b",
    r"(?i)\bcompetitive\s+conditions\b",
]

PRODUCT_SERVICE_HEADING_PATTERNS: list[str] = [
    r"(?mi)^\s*(?:our\s+)?(?:products?\s*(?:and\s+services?)?|services?\s*(?:and\s+products?)?)\s*$",
    r"(?mi)^\s*(?:product|service)\s+overview\s*$",
    r"(?mi)^\s*(?:product|service)\s+(?:lines?|portfolio|offerings?)\s*$",
    r"(?mi)^\s*(?:solutions?|platform|technology)\s+overview\s*$",
]

GEOGRAPHY_HEADING_PATTERNS: list[str] = [
    r"(?mi)^\s*(?:geographic|geography|international|global)\s+(?:overview|presence|reach|distribution|operations?)\s*$",
    r"(?mi)^\s*(?:sales\s+)?channels?\s*$",
    r"(?mi)^\s*distribution\s*$",
    r"(?mi)^\s*international\s+(?:operations?|markets?)\s*$",
]

CUSTOMER_HEADING_PATTERNS: list[str] = [
    r"(?mi)^\s*(?:our\s+)?customers?\s*$",
    r"(?mi)^\s*(?:customer|client)\s+(?:base|segments?|overview)\s*$",
    r"(?mi)^\s*(?:end[\s-]?users?|use\s+cases?)\s*$",
    r"(?mi)^\s*(?:target\s+market|markets\s+served)\s*$",
]

# Heuristic keywords for future_profile_hint
_GEOGRAPHY_KEYWORDS = re.compile(
    r"\b(?:united\s+states|u\.s\.|europe|asia|latin\s+america|canada|china|india|"
    r"japan|global|international|domestic|region|country|countries|worldwide|abroad)\b",
    re.I,
)
_CUSTOMER_KEYWORDS = re.compile(
    r"\b(?:customers?|clients?|end[\s-]?users?|buyers?|enterprises?|consumers?|subscribers?)\b",
    re.I,
)
_PRODUCT_KEYWORDS = re.compile(
    r"\b(?:products?|services?|solutions?|platforms?|software|hardware|offerings?|portfolio)\b",
    re.I,
)
_SEGMENT_KEYWORDS = re.compile(
    r"\b(?:segments?|verticals?|divisions?|business\s+units?|subsidiaries?|brands?)\b",
    re.I,
)


def _compile(patterns: list[str], flags: int = re.I) -> list[tuple[str, re.Pattern]]:
    return [(p, re.compile(p, flags)) for p in patterns]


STRICT_COMPILED = _compile(STRICT_EXPLICIT_PATTERNS)
CONTEXTUAL_COMPILED = _compile(CONTEXTUAL_EXPLICIT_PATTERNS)
IMPLICIT_COMPILED = _compile(IMPLICIT_OR_BROAD_PATTERNS)
DEMOTION_COMPILED = [re.compile(p, re.I) for p in DEMOTION_PATTERNS]
COMPETITION_HEADING_COMPILED = [re.compile(p) for p in COMPETITION_HEADING_PATTERNS]
PRODUCT_HEADING_COMPILED = [re.compile(p) for p in PRODUCT_SERVICE_HEADING_PATTERNS]
GEOGRAPHY_HEADING_COMPILED = [re.compile(p) for p in GEOGRAPHY_HEADING_PATTERNS]
CUSTOMER_HEADING_COMPILED = [re.compile(p) for p in CUSTOMER_HEADING_PATTERNS]
HEADING_FALLBACK_COMPILED = [re.compile(p) for p in HEADING_FALLBACK_PATTERNS]


def classify_section_type(sentences: list[str]) -> str:
    """Classify document section presence from headings/content cues."""
    has_competition = False
    has_business = False

    for s in sentences:
        if _match_heading_patterns(s, COMPETITION_HEADING_COMPILED):
            has_competition = True
        if re.search(r"(?mi)^\s*item\s*1\s*[.\-:\u2013\u2014]?\s*business\b", s):
            has_business = True
        elif re.search(r"(?mi)^\s*(our\s+)?business\b", s):
            has_business = True

        if has_competition and has_business:
            return "both"

    if has_competition:
        return "competition_only"
    if has_business:
        return "business_only"
    return "neither"


# =============================================================================
# Sentence splitting
# =============================================================================

_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd|Co|St|Ave|Blvd|"
    r"approx|est|fig|no|vol|e\.g|i\.e|U\.S|U\.K|p\.m|a\.m)\.",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    if not text or not text.strip():
        return []
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=[.!?])\n+|\n{2,}",
        protected,
    )
    sentences = []
    for part in parts:
        restored = part.replace("\x00", ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


# =============================================================================
# Heading detection
# =============================================================================

def _is_heading_line(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 120:
        return False
    if re.search(r"[a-z]\.[A-Z]", s):
        return False
    return (
        (len(s) <= 80 and not s.endswith("."))
        or (s.isupper() and len(s) >= 3)
        or bool(re.match(r"^[A-Z][^a-z]*(?:\s+[A-Z][^\s]*)*\s*:?\s*$", s) and len(s) < 80)
        or s.endswith(":")
    )


def _match_heading_patterns(sentence: str, patterns: list) -> bool:
    return any(p.search(sentence) for p in patterns)


def detect_local_heading_category(sentences: list[str], cue_idx: int, lookback: int = 20) -> str:
    start = max(0, cue_idx - lookback)
    for i in range(cue_idx, start - 1, -1):
        s = sentences[i]
        if not _is_heading_line(s):
            continue
        if _match_heading_patterns(s, COMPETITION_HEADING_COMPILED):
            return "competition_heading"
        if _match_heading_patterns(s, PRODUCT_HEADING_COMPILED):
            return "product_service_heading"
        if _match_heading_patterns(s, GEOGRAPHY_HEADING_COMPILED):
            return "geography_channel_heading"
        if _match_heading_patterns(s, CUSTOMER_HEADING_COMPILED):
            return "customer_heading"
        return "general_business_heading"
    return "no_heading"


def get_nearest_heading_text(sentences: list[str], cue_idx: int, lookback: int = 20) -> str:
    start = max(0, cue_idx - lookback)
    for i in range(cue_idx, start - 1, -1):
        s = sentences[i].strip()
        if _is_heading_line(s):
            return s
    return ""


def is_under_competition_heading(sentences: list[str], cue_idx: int, lookback: int = 20) -> bool:
    start = max(0, cue_idx - lookback)
    for i in range(start, cue_idx + 1):
        if _match_heading_patterns(sentences[i], COMPETITION_HEADING_COMPILED):
            return True
    return False


def is_competition_fallback_heading(sentence: str) -> bool:
    """True for clearly competition-related headings used by fallback windows only."""
    if not _is_heading_line(sentence):
        return False
    return any(p.search(sentence) for p in HEADING_FALLBACK_COMPILED)


def iter_heading_blocks(sentences: list[str]) -> Iterator[tuple[int, int, str]]:
    """
    Yield heading-scoped blocks as (heading_idx, block_end_idx, heading_text).
    block_end_idx is the sentence index before the next heading, inclusive.
    """
    heading_indices = [i for i, s in enumerate(sentences) if _is_heading_line(s)]
    if not heading_indices:
        return
    for pos, heading_idx in enumerate(heading_indices):
        next_heading_idx = heading_indices[pos + 1] if pos + 1 < len(heading_indices) else len(sentences)
        yield heading_idx, next_heading_idx - 1, sentences[heading_idx].strip()


def has_strict_cue_nearby(sentences: list[str], cue_idx: int, radius: int = 2) -> bool:
    """True if a strict-explicit cue appears within radius sentences of cue_idx."""
    start = max(0, cue_idx - radius)
    end = min(len(sentences), cue_idx + radius + 1)
    for i in range(start, end):
        if i == cue_idx:
            continue
        if any(pat.search(sentences[i]) for _, pat in STRICT_COMPILED):
            return True
    return False


# =============================================================================
# Demotion check: returns (reason_string or None, should_suppress)
# =============================================================================

def check_demotion(sentence: str, window_text: str) -> tuple[str | None, bool]:
    """
    If sentence/window matches demotion context, return (reason, suppress).
    suppress=True means do not emit a window; False means emit with demotion_reason set.
    """
    for pat in DEMOTION_COMPILED:
        if pat.search(sentence):
            p = pat.pattern
            if "customers?" in p or "clients?" in p and "include" in p:
                return ("customer_example", True)
            if "partners?" in p or "suppliers?" in p or "vendors?" in p:
                return ("partner_vendor_mention", True)
            if "examples?" in p and "include" in p or "for" in p and "example" in p:
                return ("example_language", True)
            if "regulatory" in p or "patent" in p or "compliance" in p:
                return ("regulation_compliance", True)
            if "product" in p and ("lines?" in p or "portfolio" in p) or "platform" in p:
                return ("product_catalog", True)
            if "competitive" in p and ("advantage" in p or "position" in p):
                return ("generic_competitive_advantage", False)
            return ("negative_context", True)
    return (None, False)


# =============================================================================
# Cue detection: returns (cue_group, cue_text, cue_tier, base_priority, demotion_reason) or None
# =============================================================================

def find_cue(sentence: str) -> tuple[str, str, int, int, str | None] | None:
    """
    Find best-matching cue. Order: strict > contextual > implicit.
    Strict explicit cues always win; demotion only suppresses contextual/implicit.
    Contextual is upgraded later when under competition heading or strict cue nearby.
    """
    demotion_reason, suppress = check_demotion(sentence, sentence)

    for _, pat in STRICT_COMPILED:
        m = pat.search(sentence)
        if m:
            return (CUE_GROUP_STRICT, m.group(0), 4, PRIORITY_STRICT, demotion_reason)

    for _, pat in CONTEXTUAL_COMPILED:
        m = pat.search(sentence)
        if m:
            if suppress:
                return None
            return (CUE_GROUP_CONTEXTUAL, m.group(0), 3, PRIORITY_CONTEXTUAL, demotion_reason)

    if suppress or demotion_reason:
        return None

    for _, pat in IMPLICIT_COMPILED:
        m = pat.search(sentence)
        if m:
            return (CUE_GROUP_IMPLICIT, m.group(0), 2, PRIORITY_IMPLICIT, None)

    return None


# =============================================================================
# Window computation
# =============================================================================

def compute_window(
    sentences: list[str],
    cue_idx: int,
    cue_group: str,
    base_priority: int,
    under_competition_heading: bool,
    strict_nearby: bool,
) -> tuple[int, int, int]:
    """Return (start_idx, end_idx, final_priority)."""
    priority = base_priority

    if cue_group == CUE_GROUP_STRICT:
        if under_competition_heading:
            prev_exp, next_exp = WINDOW_STRICT_UNDER_HEADING
            priority += PRIORITY_COMPETITION_HEADING_BONUS
        else:
            prev_exp, next_exp = WINDOW_STRICT
    elif cue_group == CUE_GROUP_CONTEXTUAL:
        prev_exp, next_exp = WINDOW_CONTEXTUAL
        if under_competition_heading or strict_nearby:
            priority = PRIORITY_CONTEXTUAL_UPGRADED
            prev_exp = min(prev_exp + WINDOW_HEADING_EXTRA, 2)
            next_exp = min(next_exp + WINDOW_HEADING_EXTRA, 2)
    else:
        prev_exp, next_exp = WINDOW_IMPLICIT

    prev_exp = min(prev_exp, WINDOW_MAX_SENTENCES // 2)
    next_exp = min(next_exp, WINDOW_MAX_SENTENCES // 2)
    start = max(0, cue_idx - prev_exp)
    end = min(len(sentences) - 1, cue_idx + next_exp)
    return start, end, priority


# =============================================================================
# future_profile_hint
# =============================================================================

def infer_future_profile_hint(
    window_text: str,
    local_heading_category: str,
    cue_group: str,
) -> str:
    if local_heading_category == "competition_heading":
        return "competition_market"
    if local_heading_category == "geography_channel_heading":
        return "geography_channel"
    if local_heading_category == "customer_heading":
        return "customer_use_case"
    if local_heading_category == "product_service_heading":
        return "products_services"
    if _GEOGRAPHY_KEYWORDS.search(window_text):
        return "geography_channel"
    if _CUSTOMER_KEYWORDS.search(window_text) and cue_group != CUE_GROUP_IMPLICIT:
        return "customer_use_case"
    if _SEGMENT_KEYWORDS.search(window_text):
        return "segment"
    if _PRODUCT_KEYWORDS.search(window_text) and cue_group == CUE_GROUP_IMPLICIT:
        return "products_services"
    if cue_group in (CUE_GROUP_STRICT, CUE_GROUP_CONTEXTUAL):
        return "competition_market"
    return "unknown"


# =============================================================================
# Manifest: only files resolved from manifest are processed
# =============================================================================

def load_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_filename_metadata(filename: str, source_path: str = "") -> dict[str, str]:
    stem = Path(filename).stem
    if stem.endswith("_business"):
        stem = stem[:-9]
    m = re.match(r"(\d{4}-\d{2}-\d{2})__(.+)__(.+)", stem)
    out: dict[str, str] = {
        "filing_date": "",
        "source_company_name": "",
        "accession_number": "",
        "filing_year": "",
        "source_cik": "",
        "submitter_cik": "",
    }
    if m:
        out["filing_date"] = m.group(1)
        out["source_company_name"] = m.group(2).strip()
        out["accession_number"] = m.group(3).strip()
        out["filing_year"] = m.group(1)[:4]
    acc_match = re.search(r"(\d{10})-\d{2}-\d{6}", out["accession_number"])
    if acc_match:
        out["submitter_cik"] = str(int(acc_match.group(1)))

    # Conservative fallback only when path clearly encodes issuer CIK directory,
    # e.g. .../SEC/2024_10K_raw/<issuer_cik>/<file>.txt
    if source_path:
        path_parts = Path(source_path).parts
        for idx, part in enumerate(path_parts[:-1]):
            if part.endswith("_10K_raw") and idx + 1 < len(path_parts):
                cik_candidate = path_parts[idx + 1]
                if cik_candidate.isdigit():
                    out["source_cik"] = str(int(cik_candidate))
                break
    return out


def normalize_manifest_row(row: dict[str, Any], base_dir: Path) -> dict[str, Any] | None:
    """
    Resolve the text file path from a manifest row only. No directory scanning.
    Only manifest-provided paths are used; unrelated .txt files are never included.
    """
    file_path_str = (
        row.get("business_file")
        or row.get("business_path")
        or row.get("file_path")
        or row.get("extracted_file")
        or row.get("cleaned_file")
        or row.get("cleaned_path")
        or row.get("source_file")
        or row.get("source_path")
    )
    if not file_path_str:
        return None

    path = Path(str(file_path_str).replace("\\", "/"))
    if not path.is_absolute():
        path = base_dir / path

    source_path_str = str(row.get("source_file") or row.get("source_path") or "")
    meta = _parse_filename_metadata(path.name, source_path_str)
    for key in (
        "source_company_name",
        "source_cik",
        "submitter_cik",
        "filing_year",
        "filing_date",
        "accession_number",
    ):
        val = str(row.get(key) or "").strip()
        if val:
            meta[key] = val
    section_type = str(row.get("section_type") or "business").strip() or "business"

    return {
        "file_path": path,
        "source_filename": path.name,
        "source_company_name": meta["source_company_name"],
        "source_cik": meta["source_cik"],
        "submitter_cik": meta["submitter_cik"],
        "filing_year": meta["filing_year"],
        "filing_date": meta["filing_date"],
        "accession_number": meta["accession_number"],
        "section_type": section_type,
    }


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


# =============================================================================
# Overlap-aware deduplication
# =============================================================================

def _overlap_ratio(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap on character bigrams to avoid tiny wording differences."""
    if not text_a or not text_b:
        return 0.0
    a = set(text_a[i : i + 2] for i in range(len(text_a) - 1))
    b = set(text_b[i : i + 2] for i in range(len(text_b) - 1))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def deduplicate_windows(windows: list[dict[str, Any]], threshold: float = OVERLAP_THRESHOLD) -> list[dict[str, Any]]:
    """
    When two windows (same source file) have overlap >= threshold, keep the one with
    higher window_priority, then higher cue_tier. Dropped windows are removed.
    Kept window gets is_deduplicated=True if it replaced an overlapping duplicate.
    """
    if not windows:
        return []
    key_fn = lambda w: (w.get("source_filename", ""), -w.get("window_priority", 0), -w.get("cue_tier", 0))
    sorted_w = sorted(windows, key=key_fn)
    kept: list[dict[str, Any]] = []
    for w in sorted_w:
        w = dict(w)
        w["is_deduplicated"] = False
        overlap_with = None
        for k in kept:
            if k.get("source_filename") != w.get("source_filename"):
                continue
            if _overlap_ratio(w.get("window_text", ""), k.get("window_text", "")) >= threshold:
                overlap_with = k
                break
        if overlap_with is not None:
            overlap_with["is_deduplicated"] = True
            continue
        kept.append(w)
    return kept


# =============================================================================
# Per-file processing
# =============================================================================

def process_file(entry: dict[str, Any], window_id_start: int) -> tuple[list[dict[str, Any]], int]:
    """Process one file; only paths from manifest. Returns (windows, next_id)."""
    path: Path = entry["file_path"]
    if not path.exists():
        return [], window_id_start

    text = read_text(path)
    sentences = split_sentences(text)
    if not sentences:
        return [], window_id_start

    analyzed_section_type = classify_section_type(sentences)

    raw: list[dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        cue_result = find_cue(sent)
        if cue_result is None:
            continue

        cue_group, cue_text, cue_tier, base_priority, demotion_reason = cue_result
        under_heading = is_under_competition_heading(sentences, idx)
        strict_nearby = has_strict_cue_nearby(sentences, idx)
        start, end, final_priority = compute_window(
            sentences, idx, cue_group, base_priority, under_heading, strict_nearby
        )

        local_heading_category = detect_local_heading_category(sentences, idx)
        heading_text = get_nearest_heading_text(sentences, idx)
        window_sentences = sentences[start : end + 1]
        window_text = " ".join(window_sentences)
        hint = infer_future_profile_hint(window_text, local_heading_category, cue_group)

        raw.append({
            "trigger_sentence_idx": idx,
            "start_sentence_idx": start,
            "end_sentence_idx": end,
            "cue_group": cue_group,
            "cue_tier": cue_tier,
            "cue_text": cue_text,
            "window_priority": final_priority,
            "heading_text": heading_text,
            "local_heading_category": local_heading_category,
            "trigger_sentence": sent,
            "window_text": window_text,
            "text_char_count": len(window_text),
            "future_profile_hint": hint,
            "demotion_reason": demotion_reason or "",
            "fallback_reason": "",
            "is_heading_fallback": False,
        })

    # Heading-triggered fallback: one very low-priority broad window per
    # competition-related heading block that otherwise has no cue-based windows.
    blocks_with_regular_windows = set()
    for rec in raw:
        for heading_idx, block_end, _heading_text in iter_heading_blocks(sentences):
            body_start = heading_idx + 1
            if body_start <= rec["trigger_sentence_idx"] <= block_end:
                blocks_with_regular_windows.add(heading_idx)
                break

    for heading_idx, block_end, heading_text in iter_heading_blocks(sentences):
        if heading_idx in blocks_with_regular_windows:
            continue
        if not is_competition_fallback_heading(heading_text):
            continue

        fallback_idx = None
        for i in range(heading_idx + 1, block_end + 1):
            s = sentences[i].strip()
            if not s or _is_heading_line(s):
                continue
            if not re.search(r"\w", s):
                continue
            fallback_idx = i
            break

        if fallback_idx is None:
            continue

        fallback_text = sentences[fallback_idx].strip()
        raw.append({
            "trigger_sentence_idx": fallback_idx,
            "start_sentence_idx": fallback_idx,
            "end_sentence_idx": fallback_idx,
            "cue_group": CUE_GROUP_HEADING_FALLBACK,
            "cue_tier": 1,
            "cue_text": heading_text,
            "window_priority": PRIORITY_HEADING_FALLBACK,
            "heading_text": heading_text,
            "local_heading_category": "competition_heading",
            "trigger_sentence": fallback_text,
            "window_text": fallback_text,
            "text_char_count": len(fallback_text),
            "future_profile_hint": "competition_market",
            "demotion_reason": "",
            "fallback_reason": "competition_heading_without_explicit_cue",
            "is_heading_fallback": True,
        })

    # In-file dedup by (start, end): keep highest priority
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in raw:
        key = (rec["start_sentence_idx"], rec["end_sentence_idx"])
        if key not in seen or rec["window_priority"] > seen[key]["window_priority"]:
            seen[key] = rec

    records = []
    for rec in sorted(seen.values(), key=lambda r: r["trigger_sentence_idx"]):
        wid = f"W{window_id_start:07d}"
        window_id_start += 1
        rec_copy = dict(rec)
        rec_copy["is_deduplicated"] = False
        export_bucket = export_bucket_for_cue_group(rec_copy["cue_group"])
        records.append({
            "window_id": wid,
            "source_company_name": entry["source_company_name"],
            "source_cik": entry["source_cik"],
            "submitter_cik": entry.get("submitter_cik", ""),
            "filing_year": entry["filing_year"],
            "filing_date": entry["filing_date"],
            "accession_number": entry["accession_number"],
            "section_type": analyzed_section_type,
            "source_filename": entry["source_filename"],
            "heading_text": rec_copy["heading_text"],
            "local_heading_category": rec_copy["local_heading_category"],
            "cue_text": rec_copy["cue_text"],
            "cue_group": rec_copy["cue_group"],
            "export_bucket": export_bucket,
            "cue_tier": rec_copy["cue_tier"],
            "window_priority": rec_copy["window_priority"],
            "start_sentence_idx": rec_copy["start_sentence_idx"],
            "end_sentence_idx": rec_copy["end_sentence_idx"],
            "trigger_sentence": rec_copy["trigger_sentence"],
            "window_text": rec_copy["window_text"],
            "text_char_count": rec_copy["text_char_count"],
            "future_profile_hint": rec_copy["future_profile_hint"],
            "is_deduplicated": rec_copy["is_deduplicated"],
            "demotion_reason": rec_copy["demotion_reason"],
            "fallback_reason": rec_copy["fallback_reason"],
            "is_heading_fallback": rec_copy["is_heading_fallback"],
        })

    return records, window_id_start


# =============================================================================
# Output fields
# =============================================================================

OUTPUT_FIELDS = [
    "window_id",
    "source_company_name",
    "source_cik",
    "submitter_cik",
    "filing_year",
    "filing_date",
    "accession_number",
    "section_type",
    "source_filename",
    "heading_text",
    "local_heading_category",
    "cue_text",
    "cue_group",
    "export_bucket",
    "cue_tier",
    "window_priority",
    "start_sentence_idx",
    "end_sentence_idx",
    "trigger_sentence",
    "window_text",
    "text_char_count",
    "future_profile_hint",
    "is_deduplicated",
    "demotion_reason",
    "fallback_reason",
    "is_heading_fallback",
]


def export_bucket_for_cue_group(cue_group: str) -> str:
    """Map internal cue groups to external export buckets."""
    if cue_group == CUE_GROUP_STRICT:
        return EXPORT_BUCKET_STRICT
    if cue_group == CUE_GROUP_CONTEXTUAL:
        return EXPORT_BUCKET_CONTEXTUAL
    return EXPORT_BUCKET_BROAD


def _bucket_output_path(base_path: Path, bucket: str) -> Path:
    """Return bucket-specific output path derived from a base output path."""
    return base_path.with_name(f"{base_path.stem}_{bucket}{base_path.suffix}")


def _write_bucket_files(
    windows_by_bucket: dict[str, list[dict[str, Any]]],
    output_csv_base: Path,
    output_jsonl_base: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Write one CSV and one JSONL per export bucket."""
    csv_paths: dict[str, Path] = {}
    jsonl_paths: dict[str, Path] = {}
    ordered_buckets = [EXPORT_BUCKET_STRICT, EXPORT_BUCKET_CONTEXTUAL, EXPORT_BUCKET_BROAD]

    for bucket in ordered_buckets:
        bucket_windows = windows_by_bucket.get(bucket, [])

        csv_path = _bucket_output_path(output_csv_base, bucket)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(bucket_windows)
        csv_paths[bucket] = csv_path

        jsonl_path = _bucket_output_path(output_jsonl_base, bucket)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for w in bucket_windows:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        jsonl_paths[bucket] = jsonl_path

    return csv_paths, jsonl_paths


def _print_progress(current: int, total: int, prefix: str = "Processing") -> None:
    """Render a simple in-place progress bar for long runs."""
    if total <= 0:
        return
    width = 32
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(ratio * 100)
    end = "\n" if current >= total else ""
    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total} ({pct}%)")
    sys.stdout.flush()
    if end:
        sys.stdout.write(end)


def _default_output_dir(manifest_path: Path) -> Path:
    """Choose a year-specific default output folder when possible."""
    parent = manifest_path.parent
    candidates = [manifest_path.name, parent.name]
    for candidate in candidates:
        m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", candidate)
        if m:
            return (parent.parent / f"{m.group(1)}_abcd").resolve()
    return (Path(".").resolve() / "abcd_output").resolve()


# =============================================================================
# Audit summary: only intended audit text, no stray content
# =============================================================================

def build_audit_summary(
    total_files: int,
    files_with_text: int,
    files_with_windows: int,
    zero_window_files: list[str],
    windows_before_dedup: int,
    windows_after_dedup: int,
    all_windows: list[dict[str, Any]],
) -> str:
    """Produce the window_audit_summary.txt content. No print statements; pure string."""
    tier_counts = collections.Counter(w.get("cue_tier", 0) for w in all_windows)
    group_counts = collections.Counter(w.get("cue_group", "") for w in all_windows)
    section_counts = collections.Counter(w.get("section_type", "") for w in all_windows)
    hint_counts = collections.Counter(w.get("future_profile_hint", "") for w in all_windows)
    priority_counts = collections.Counter(w.get("window_priority", 0) for w in all_windows)
    cue_counts = collections.Counter(w.get("cue_text", "").lower() for w in all_windows)
    demotion_counts = collections.Counter(w.get("demotion_reason", "") for w in all_windows if w.get("demotion_reason"))
    fallback_windows = [w for w in all_windows if w.get("is_heading_fallback")]
    heading_fallback_count = len(fallback_windows)

    windows_by_file: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for w in all_windows:
        windows_by_file[w.get("source_filename", "")].append(w)
    rescued_files = [
        fname
        for fname, ws in windows_by_file.items()
        if any(w.get("is_heading_fallback") for w in ws) and not any(not w.get("is_heading_fallback") for w in ws)
    ]

    top_cues = cue_counts.most_common(20)
    top_demotions = demotion_counts.most_common(10)
    total_chars = sum(w.get("text_char_count", 0) for w in all_windows)
    avg_len = (total_chars / len(all_windows)) if all_windows else 0

    lines = [
        "=" * 72,
        "CANDIDATE WINDOWS — AUDIT SUMMARY",
        "=" * 72,
        "",
        "── FILE COUNTS ───────────────────────────────────────────────────────",
        f"  Files in manifest (processed)   : {total_files:,}",
        f"  Files with resolved text path   : {files_with_text:,}",
        f"  Files with at least one window  : {files_with_windows:,}",
        f"  Files with zero windows         : {len(zero_window_files):,}",
        "",
        "── WINDOW COUNTS ──────────────────────────────────────────────────────",
        f"  Total windows (before dedup)    : {windows_before_dedup:,}",
        f"  Total windows (after dedup)     : {windows_after_dedup:,}",
        f"  Heading fallback windows        : {heading_fallback_count:,}",
        f"  Files rescued by fallback       : {len(rescued_files):,}",
        "",
        "  By cue_group:",
    ]
    cue_group_order = [
        CUE_GROUP_STRICT,
        CUE_GROUP_CONTEXTUAL,
        CUE_GROUP_IMPLICIT,
        CUE_GROUP_DEMOTION,
        CUE_GROUP_HEADING_FALLBACK,
    ]
    for grp in cue_group_order:
        if grp in group_counts:
            lines.append(f"    {grp:<35s} : {group_counts[grp]:,}")
    for grp in sorted(group_counts.keys()):
        if grp in cue_group_order:
            continue
        lines.append(f"    {grp:<35s} : {group_counts[grp]:,}")

    lines += [
        "",
        "  By cue_tier:",
    ]
    for t in sorted(tier_counts.keys()):
        lines.append(f"    Tier {t:<30} : {tier_counts[t]:,}")

    lines += [
        "",
        "  By window_priority:",
    ]
    for p in sorted(priority_counts.keys(), reverse=True):
        lines.append(f"    Priority {p:<28} : {priority_counts[p]:,}")

    lines += [
        "",
        "  By section_type:",
    ]
    for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {sec:<35s} : {cnt:,}")

    lines += [
        "",
        "  By future_profile_hint:",
    ]
    for hint, cnt in sorted(hint_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {hint:<35s} : {cnt:,}")

    lines += [
        "",
        "── WINDOW LENGTH ───────────────────────────────────────────────────────",
        f"  Average window char length      : {avg_len:,.1f}",
        "",
        "── TOP 20 CUE PHRASES ─────────────────────────────────────────────────",
    ]
    for rank, (cue, cnt) in enumerate(top_cues, 1):
        lines.append(f"  {rank:>2}. {cue:<42s} {cnt:,}")

    if top_demotions:
        lines += [
            "",
            "── TOP DEMOTION REASONS ─────────────────────────────────────────────",
        ]
        for reason, cnt in top_demotions:
            lines.append(f"  {reason or '(empty)':<40s} {cnt:,}")

    if zero_window_files:
        lines += [
            "",
            "── FILES WITH ZERO WINDOWS ─────────────────────────────────────────",
        ]
        for fname in sorted(zero_window_files)[:50]:
            lines.append(f"  {fname}")
        if len(zero_window_files) > 50:
            lines.append(f"  ... and {len(zero_window_files) - 50} more")

    if rescued_files:
        lines += [
            "",
            "── FILES RESCUED BY HEADING FALLBACK ───────────────────────────────",
        ]
        for fname in sorted(rescued_files)[:50]:
            lines.append(f"  {fname}")
        if len(rescued_files) > 50:
            lines.append(f"  ... and {len(rescued_files) - 50} more")

    lines += [
        "",
        "── SAMPLE HEADING FALLBACK WINDOWS ───────────────────────────────────",
    ]
    if not fallback_windows:
        lines.append("  (none)")
    else:
        for w in fallback_windows[:5]:
            trigger = (w.get("trigger_sentence") or "")[:100]
            company = w.get("source_company_name", "")
            heading = w.get("heading_text", "")
            lines.append(
                f"  {w.get('window_id', '')} | {company} | heading: {heading} | trigger: {trigger}..."
            )

    lines += ["", "=" * 72]
    return "\n".join(lines)


# =============================================================================
# Main run
# =============================================================================

def run(
    manifest_path: Path,
    base_dir: Path,
    output_csv: Path,
    output_jsonl: Path,
    output_audit: Path,
    limit: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Load manifest, process only manifest-resolved files, deduplicate, write outputs."""
    manifest_rows = load_manifest(manifest_path)
    base_dir = base_dir.resolve()

    entries = []
    for row in manifest_rows:
        norm = normalize_manifest_row(row, base_dir)
        if norm:
            entries.append(norm)

    if limit is not None:
        entries = entries[:limit]

    all_windows = []
    files_with_windows = set()
    zero_window_files = []
    window_id_counter = 1

    total_entries = len(entries)
    iter_entries = entries
    if show_progress and tqdm is not None:
        iter_entries = tqdm(entries, desc="ABCD file scan", unit="file")

    for idx, entry in enumerate(iter_entries, 1):
        records, window_id_counter = process_file(entry, window_id_counter)
        if records:
            all_windows.extend(records)
            files_with_windows.add(entry["source_filename"])
        elif entry["file_path"].exists():
            zero_window_files.append(entry["source_filename"])
        if show_progress and tqdm is None:
            _print_progress(idx, total_entries, prefix="ABCD file scan")

    windows_before_dedup = len(all_windows)
    all_windows = deduplicate_windows(all_windows)
    windows_after_dedup = len(all_windows)

    all_windows.sort(
        key=lambda w: (w.get("filing_year", ""), w.get("source_company_name", ""), -w.get("window_priority", 0))
    )

    windows_by_bucket: dict[str, list[dict[str, Any]]] = {
        EXPORT_BUCKET_STRICT: [],
        EXPORT_BUCKET_CONTEXTUAL: [],
        EXPORT_BUCKET_BROAD: [],
    }
    for w in all_windows:
        bucket = str(w.get("export_bucket") or export_bucket_for_cue_group(str(w.get("cue_group", ""))))
        if bucket not in windows_by_bucket:
            bucket = EXPORT_BUCKET_BROAD
        windows_by_bucket[bucket].append(w)

    bucket_csv_paths, bucket_jsonl_paths = _write_bucket_files(windows_by_bucket, output_csv, output_jsonl)

    summary_text = build_audit_summary(
        total_files=len(manifest_rows),
        files_with_text=len(entries),
        files_with_windows=len(files_with_windows),
        zero_window_files=zero_window_files,
        windows_before_dedup=windows_before_dedup,
        windows_after_dedup=windows_after_dedup,
        all_windows=all_windows,
    )
    output_audit.write_text(summary_text, encoding="utf-8")

    return {
        "total_manifest_rows": len(manifest_rows),
        "files_with_text": len(entries),
        "files_with_windows": len(files_with_windows),
        "total_windows_before_dedup": windows_before_dedup,
        "total_windows_after_dedup": windows_after_dedup,
        "bucket_counts": {k: len(v) for k, v in windows_by_bucket.items()},
        "bucket_csv_paths": {k: str(v) for k, v in bucket_csv_paths.items()},
        "bucket_jsonl_paths": {k: str(v) for k, v in bucket_jsonl_paths.items()},
        "zero_window_files": len(zero_window_files),
        "summary_text": summary_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Build cue-based candidate windows for explicit competition evidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python build_candidate_windows.py manifest.json
              python build_candidate_windows.py manifest.csv --base-dir 2023_10K_business
              python build_candidate_windows.py manifest.json --limit 50 --output-dir results/
        """),
    )
    parser.add_argument("manifest", type=Path, help="Path to manifest JSON or CSV.")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base dir for resolving relative paths.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for output files.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--output-audit", type=Path, default=None, help="Output audit summary path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N manifest entries.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress bar output.")
    args = parser.parse_args()

    out_dir = (args.output_dir or _default_output_dir(args.manifest)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or (out_dir / "candidate_windows.csv")
    output_jsonl = args.output_jsonl or (out_dir / "candidate_windows.jsonl")
    output_audit = args.output_audit or (out_dir / "window_audit_summary.txt")

    stats = run(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        output_audit=output_audit,
        limit=args.limit,
        show_progress=not args.no_progress,
    )

    print("\n" + stats["summary_text"])
    print("\nOutputs written by export bucket:")
    for bucket in [EXPORT_BUCKET_STRICT, EXPORT_BUCKET_CONTEXTUAL, EXPORT_BUCKET_BROAD]:
        count = stats["bucket_counts"].get(bucket, 0)
        csv_path = stats["bucket_csv_paths"].get(bucket, "")
        jsonl_path = stats["bucket_jsonl_paths"].get(bucket, "")
        print(f"  [{bucket}] {count} windows")
        print(f"    CSV:   {csv_path}")
        print(f"    JSONL: {jsonl_path}")
    print(f"  {output_audit}")


if __name__ == "__main__":
    main()