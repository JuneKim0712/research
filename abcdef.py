#!/usr/bin/env python3
"""
Raw organization / Organization   mention harvest from explicit_candidate_windows.csv.

Runs two off-the-shelf extractors on a chosen text column per row. Default ``--ner-text-column auto``:
if the CSV has ``org_mentions_union_filtered``, NER runs only on that string; otherwise ``window_text``.
Recall-first union:
  1) spaCy English NER — keep spans labeled ORG, GPE/LOC (locations), and PRODUCT.
  2) Hugging Face token-classification NER (default: Jean-Baptiste/roberta-large-ner-english, RoBERTa on CoNLL-2003) —
     by default keep ORG, LOC, MISC, and PRODUCT if the model emits them. MISC is mapped to coarse attribute ORG.
     Standard CoNLL models have no PRODUCT label; add it only after fine-tuning a model that predicts PRODUCT.

Same (window_id, char span in *original* NER source text for that row) from both models is merged into one row
with extractor_name like "roberta_ner+spacy" (HF tag set by --hf-extractor-tag). Same mention string at different spans stays
as separate rows. Mentions in different windows are never deduplicated.

Dependencies (see requirements-explicit-mentions.txt):
  pip install -r requirements-explicit-mentions.txt
  python -m spacy download en_core_web_sm   # or en_core_web_md / en_core_web_lg for recall

Input default:  ./explicit_candidate_windows.csv
Output default: ./explicit_mentions_raw.csv (window-level span list), ./explicit_mentions_raw_nonraw.csv (deduped),
  and ./explicit_mentions_raw_org_diff.csv (per-window ORG surplus vs nonraw, unless --org-diff-log is set).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# --- Optional heavy imports (clear error if missing) -------------------------------------------
try:
    import spacy
except Exception as e:  # pragma: no cover
    spacy = None  # type: ignore
    _SPACY_IMPORT_ERROR = e
else:
    _SPACY_IMPORT_ERROR = None

try:
    from transformers import pipeline
except ImportError as e:  # pragma: no cover
    pipeline = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR = e
else:
    _TRANSFORMERS_IMPORT_ERROR = None

# Default HF model: RoBERTa-large CoNLL English (ORG / LOC / PER / MISC; PRODUCT only if the checkpoint defines it).
DEFAULT_HF_NER_MODEL = "Jean-Baptiste/roberta-large-ner-english"
DEFAULT_HF_EXTRACTOR_TAG = "roberta_ner"
DEFAULT_HF_TYPE_MODEL = "facebook/bart-large-mnli"
# Comma-separated default for --hf-keep-labels (parsed below).
DEFAULT_HF_NER_LABELS_STR = "ORG,LOC,MISC,PRODUCT"


def parse_hf_ner_labels(s: str) -> frozenset[str]:
    """Parse comma/whitespace-separated uppercase HF entity labels (e.g. ORG,LOC,PRODUCT)."""
    out: set[str] = set()
    for part in re.split(r"[,;\s]+", (s or "").strip().upper()):
        if part:
            out.add(part)
    return frozenset(out)


DEFAULT_HF_NER_LABELS = parse_hf_ner_labels(DEFAULT_HF_NER_LABELS_STR)


def resolve_entity_attribute(ner_labels: Iterable[str]) -> str:
    """
    Map merged token-entity labels to a coarse attribute: ORG | LOC | PRODUCT.
    MISC and unknown labels count as ORG-like. Priority: LOC > PRODUCT > ORG.
    """
    lbs = {str(x).upper().replace("B-", "").replace("I-", "") for x in ner_labels}
    if "LOC" in lbs:
        return "LOC"
    if "PRODUCT" in lbs:
        return "PRODUCT"
    return "ORG"


# Zero-width / format characters to drop (length changes — handled in mapping pass).
_ZERO_WIDTH = frozenset(
    "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\ufeff\u00ad"
)

# Mention type classification patterns
TICKER_PATTERN = re.compile(
    r"^[A-Z]{1,5}$|^[A-Z]{1,5}\.[A-Z]$",  # e.g., AAPL, BRK.A
)
TYPE_ZS_LABELS = ["ComputingProduct", "Organization  "]
TYPE_ZS_TO_MENTION = {
    "ComputingProduct": "ComputingProduct",
    "Organization  ": "ORGANIZATION  ",
}

# Pattern-based generic competitor phrase extraction
COMPETITOR_CUE_PATTERN = re.compile(
    r"(?:competitors?|companies?|providers?|platforms?|services?|marketplaces?|players?|vendors?)\s+"
    r"(?:include|consist of|comprise|such as|like)\s+([^.!?;,\n]{15,300})(?=[.!?;,\n]|$)",
    re.IGNORECASE,
)
COMPETE_PHRASE_PATTERN = re.compile(
    r"(?:we\s+)?(?:compete|competing|competing in|compete with)\s+(?:with\s+)?([^.!?;,\n]{15,300})(?=[.!?;,\n]|$)",
    re.IGNORECASE,
)

# Pattern 3: "our competitors ... are X" style phrases
COMPETITOR_ARE_PATTERN = re.compile(
    r"(?:our\s+)?(?:(?:major|primary|principal|key|direct|significant)\s+)?competitors?(?:\s+in\s+[^.!?;,\n]{1,120})?\s+"
    r"(?:are|were|remain|include)\s+([^.!?;,\n]{15,300})(?=[.!?;,\n]|$)",
    re.IGNORECASE,
)

# Pattern 4: cue near sentence end, e.g. "X are our major competitors"
END_COMPETITOR_CUE_PATTERN = re.compile(
    r"([^.!?;,\n]{15,300})\s+(?:are|were|remain)\s+(?:our\s+)?(?:(?:major|primary|principal|key|direct|significant)\s+)?competitors?\b(?=[.!?;,\n]|$)",
    re.IGNORECASE,
)


def extract_generic_competitor_phrases(text: str) -> list[tuple[int, int, str]]:
    """
    Extract generic competitor category phrases from text.
    Returns list of (start, end, mention_text) tuples.
    Looks for patterns like "competitors include X", "we compete with Y", "platforms such as Z".
    Requires phrases to be at least 15 characters to avoid noise.
    """
    out: list[tuple[int, int, str]] = []
    seen_phrases: set[str] = set()  # Avoid duplicates
    
    # Pattern 1: "competitors include X, Y, Z" style phrases
    for match in COMPETITOR_CUE_PATTERN.finditer(text):
        phrase = match.group(1).strip()
        if phrase and len(phrase) >= 15 and phrase.lower() not in seen_phrases:
            seen_phrases.add(phrase.lower())
            start = match.start(1)
            end = match.end(1)
            out.append((start, end, phrase))
    
    # Pattern 2: "we compete with X" / "compete with Y" style phrases
    for match in COMPETE_PHRASE_PATTERN.finditer(text):
        phrase = match.group(1).strip()
        if phrase and len(phrase) >= 15 and phrase.lower() not in seen_phrases:
            seen_phrases.add(phrase.lower())
            start = match.start(1)
            end = match.end(1)
            out.append((start, end, phrase))

    # Pattern 3: "our competitors ... are X" style phrases
    for match in COMPETITOR_ARE_PATTERN.finditer(text):
        phrase = match.group(1).strip()
        if phrase and len(phrase) >= 15 and phrase.lower() not in seen_phrases:
            seen_phrases.add(phrase.lower())
            start = match.start(1)
            end = match.end(1)
            out.append((start, end, phrase))

    # Pattern 4: cue near sentence end ("X are our major competitors")
    for match in END_COMPETITOR_CUE_PATTERN.finditer(text):
        phrase = match.group(1).strip()
        if phrase and len(phrase) >= 15 and phrase.lower() not in seen_phrases:
            seen_phrases.add(phrase.lower())
            start = match.start(1)
            end = match.end(1)
            out.append((start, end, phrase))
    
    return out


def _mention_ner_signals(
    text: str,
    nlp: Any | None,
    ner_pipe: Any | None,
) -> tuple[bool, str | None]:
    """
    Single spaCy + single HF pass on mention text. Returns (is_region, ner_type_hint).
    ner_type_hint is ComputingProduct, ORGANIZATION  , or None (defer to zero-shot / default).
    """
    is_region = False
    ner_type: str | None = None

    if nlp is not None:
        try:
            doc = nlp(text)
            has_region_spacy = False
            has_product = False
            has_org_spacy = False
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC", "NORP"}:
                    has_region_spacy = True
                if ent.label_ == "PRODUCT":
                    has_product = True
                elif ent.label_ == "ORG":
                    has_org_spacy = True
            if has_region_spacy:
                return True, None
            if has_product:
                ner_type = "ComputingProduct"
            elif has_org_spacy:
                ner_type = "ORGANIZATION  "
        except Exception:
            pass

    if ner_pipe is not None:
        try:
            raw = ner_pipe(text)
        except Exception:
            raw = []
        for item in raw:
            label = item.get("entity_group") or item.get("entity") or ""
            label = str(label).upper().replace("B-", "").replace("I-", "")
            if label == "LOC":
                return True, None
            if label == "MISC":
                if re.search(r"\b(?:north|south|east|west|central)\b", text, re.IGNORECASE):
                    return True, None
                if re.search(r"\b[A-Za-z]+(?:an|ian|ean|ese|ish)\b", text):
                    return True, None
            if label == "PRODUCT":
                ner_type = "ComputingProduct"
            elif label == "ORG" and ner_type is None:
                ner_type = "ORGANIZATION  "

    return False, ner_type


def classify_mention_type(
    mention_text: str,
    window_text: str = "",
    *,
    nlp: Any | None = None,
    ner_pipe: Any | None = None,
    type_classifier: Any | None = None,
    ner_signal_cache: dict[str, tuple[bool, str | None]] | None = None,
    type_cache: dict[str, str] | None = None,
) -> str:
    """Classify a mention as ORGANIZATION  , ComputingProduct, REGION, TICKER, or COMPETITOR_CATEGORY."""
    mention_clean = mention_text.strip()
    if not mention_clean:
        return "ORGANIZATION  "  # default

    # Check ticker first (strict pattern)
    if TICKER_PATTERN.match(mention_clean):
        return "TICKER"

    signal_key = mention_clean.lower()
    if ner_signal_cache is not None and signal_key in ner_signal_cache:
        is_region, ner_type_hint = ner_signal_cache[signal_key]
    else:
        is_region, ner_type_hint = _mention_ner_signals(mention_clean, nlp, ner_pipe)
        if ner_signal_cache is not None:
            ner_signal_cache[signal_key] = (is_region, ner_type_hint)
    if is_region:
        return "REGION"

    cache_key = f"{mention_clean.lower()}\n{window_text[:240].lower()}"
    if type_cache is not None and cache_key in type_cache:
        return type_cache[cache_key]

    mention_type = _classify_type_after_ner_hints(
        mention_clean,
        window_text,
        ner_type_hint=ner_type_hint,
        type_classifier=type_classifier,
    )
    if type_cache is not None:
        type_cache[cache_key] = mention_type
    return mention_type

OUTPUT_FIELDS_RAW = [
    "window_id",
    "source_company",
    "source_cik",
    "filing_year",
    "section",
    "cue_phrase",
    "cue_group",
    "trigger_sentence",
    "window_text",
    "ner_input_text",
    "org_mentions_raw_count",
    "org_mentions_raw",
    "mention_starts_raw",
    "mention_ends_raw",
    "mention_types_raw",
    "entity_attributes_raw",
]

OUTPUT_FIELDS_NONRAW = [
    "window_id",
    "source_company",
    "source_cik",
    "filing_year",
    "section",
    "cue_phrase",
    "cue_group",
    "trigger_sentence",
    "window_text",
    "ner_input_text",
    "org_mentions_count",
    "org_mentions",
    "mention_types",
    "entity_attributes",
]

# One row per surplus ORG mention vs the nonraw (deduped-by-text) view, or per nonraw-only anomaly.
OUTPUT_FIELDS_ORG_DIFF = [
    "window_id",
    "raw_org_span_count",
    "nonraw_unique_org_count",
    "raw_mention",
    "mention_start",
    "mention_end",
    "extractor_name",
    "extra_source",
    "diff_reason",
]


def normalize_with_offset_map(text: str) -> tuple[str, list[int], list[int]]:
    """
    Light cleanup: remove zero-width junk, normalize all whitespace runs to a single ASCII space.

    Returns:
        clean_text — string passed to both NER backends
        orig_starts — for each index i in clean_text, start index in original text
        orig_ends_excl — for each index i in clean_text, exclusive end index in original for that char

    Any span [cs, ce) in clean_text maps to original [orig_starts[cs], orig_ends_excl[ce - 1]).
    """
    if not text:
        return "", [], []

    # Pass 1: drop zero-width; record kept chars with their original indices.
    kept: list[tuple[int, str]] = []
    for i, c in enumerate(text):
        if c in _ZERO_WIDTH:
            continue
        kept.append((i, c))

    # Pass 2: replace each max whitespace run (in kept stream) with one space.
    out_chars: list[str] = []
    starts: list[int] = []
    ends_excl: list[int] = []
    n = len(kept)
    j = 0
    while j < n:
        orig_i, c = kept[j]
        o = ord(c)
        is_ws = c.isspace() or o == 0xA0
        if is_ws:
            run_start_orig = orig_i
            while j < n:
                oi, ch = kept[j]
                oo = ord(ch)
                if not (ch.isspace() or oo == 0xA0):
                    break
                j += 1
            out_chars.append(" ")
            starts.append(run_start_orig)
            ends_excl.append(kept[j - 1][0] + 1)
        else:
            out_chars.append(c)
            starts.append(orig_i)
            ends_excl.append(orig_i + 1)
            j += 1

    clean = "".join(out_chars)
    return clean, starts, ends_excl


def clean_span_to_orig(
    cs: int,
    ce: int,
    orig_starts: list[int],
    orig_ends_excl: list[int],
    orig_len: int,
) -> tuple[int, int] | None:
    """Map exclusive [cs, ce) in clean_text to exclusive [o0, o1) in original; None if invalid."""
    if ce <= cs or cs < 0:
        return None
    if not orig_starts or ce > len(orig_starts):
        return None
    o_start = orig_starts[cs]
    o_end = orig_ends_excl[ce - 1]
    if o_start < 0 or o_end > orig_len or o_end < o_start:
        return None
    return o_start, o_end


def _orig_slice_for_sent(
    window_orig: str,
    s0: int,
    s1: int,
    orig_starts: list[int],
    orig_ends_excl: list[int],
    sent_fallback: str,
) -> str:
    orig_len = len(window_orig)
    if s0 >= len(orig_starts):
        return sent_fallback
    s1_adj = min(s1, len(orig_starts)) - 1
    if s1_adj < s0:
        return sent_fallback
    o0 = orig_starts[s0]
    o1 = orig_ends_excl[s1_adj]
    if 0 <= o0 <= o1 <= orig_len:
        return window_orig[o0:o1]
    return sent_fallback


def sentence_for_span_orig(
    window_orig: str,
    doc: Any,
    cs: int,
    ce: int,
    orig_starts: list[int],
    orig_ends_excl: list[int],
    sent_bounds: list[tuple[int, int]] | None = None,
    sent_texts: list[str] | None = None,
) -> str:
    """Original-window substring for the spaCy sentence containing the mention (clean offsets)."""
    orig_len = len(window_orig)
    mid = (cs + ce) // 2
    if sent_bounds and sent_texts and len(sent_bounds) == len(sent_texts):
        starts_only = [b[0] for b in sent_bounds]
        i = bisect.bisect_right(starts_only, mid) - 1
        if 0 <= i < len(sent_bounds):
            s0, s1 = sent_bounds[i]
            if s0 <= mid < s1:
                return _orig_slice_for_sent(
                    window_orig, s0, s1, orig_starts, orig_ends_excl, sent_texts[i]
                )
        return ""

    for sent in doc.sents:
        if sent.start_char <= mid < sent.end_char:
            s0, s1 = sent.start_char, sent.end_char
            return _orig_slice_for_sent(
                window_orig, s0, s1, orig_starts, orig_ends_excl, sent.text
            )
    return ""


def load_spacy(model_name: str):
    if spacy is None:
        raise RuntimeError(f"spaCy not installed: {_SPACY_IMPORT_ERROR}")
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model {model_name!r} not found. Install e.g.:\n"
            f"  python -m spacy download {model_name}"
        ) from None


def load_hf_ner(model_name: str = DEFAULT_HF_NER_MODEL, device: int | str = -1):
    if pipeline is None:
        raise RuntimeError(f"transformers not installed: {_TRANSFORMERS_IMPORT_ERROR}")
    # Token-classification NER; aggregation merges subwords into word/entity spans with char positions.
    return pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )

def load_hf_type_classifier(model_name: str = DEFAULT_HF_TYPE_MODEL, device: int | str = -1):
    if pipeline is None:
        raise RuntimeError(f"transformers not installed: {_TRANSFORMERS_IMPORT_ERROR}")
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

def _classify_type_after_ner_hints(
    mention_text: str,
    window_text: str,
    *,
    ner_type_hint: str | None,
    type_classifier: Any | None,
) -> str:
    """Finish type classification using cached NER hints; zero-shot only when hints are inconclusive."""
    text = mention_text.strip()
    if not text:
        return "ORGANIZATION  "

    if ner_type_hint == "ComputingProduct":
        return "ComputingProduct"
    if ner_type_hint == "ORGANIZATION  ":
        return "ORGANIZATION  "

    if type_classifier is None:
        return "ORGANIZATION  "

    context = (window_text or "").strip()
    if len(context) > 320:
        context = context[:320]
    premise = f"Mention: {text}. Context: {context}" if context else f"Mention: {text}."
    try:
        result = type_classifier(premise, TYPE_ZS_LABELS, multi_label=False)
        labels = result.get("labels") or []
        if labels:
            top = str(labels[0]).lower()
            return TYPE_ZS_TO_MENTION.get(top, "ORGANIZATION  ")
    except Exception:
        pass
    return "ORGANIZATION  "


def extract_spacy_labeled_spans(doc: Any) -> list[tuple[int, int, str]]:
    """ORG, GPE/LOC, PRODUCT spans in *clean* text; GPE is folded to LOC for coarse attributes."""
    out: list[tuple[int, int, str]] = []
    for ent in doc.ents:
        if ent.start_char is None or ent.end_char is None:
            continue
        if ent.label_ == "ORG":
            out.append((ent.start_char, ent.end_char, "ORG"))
        elif ent.label_ in ("GPE", "LOC"):
            out.append((ent.start_char, ent.end_char, "LOC"))
        elif ent.label_ == "PRODUCT":
            out.append((ent.start_char, ent.end_char, "PRODUCT"))
    return out


def extract_hf_labeled_spans(
    ner_pipe: Any,
    clean_text: str,
    keep_labels: frozenset[str],
) -> list[tuple[int, int, str]]:
    if not clean_text.strip() or not keep_labels:
        return []
    raw = ner_pipe(clean_text)
    out: list[tuple[int, int, str]] = []
    for item in raw:
        label = item.get("entity_group") or item.get("entity") or ""
        label = str(label).upper().replace("B-", "").replace("I-", "")
        if label not in keep_labels:
            continue
        start = int(item["start"])
        end = int(item["end"])
        if end > start:
            out.append((start, end, label))
    return out


def merge_spans_for_window(
    spans: Iterable[tuple[int, int, str, int, int, str | None]],
) -> dict[tuple[int, int], tuple[set[str], set[str], int, int]]:
    """
    Key: (orig_start, orig_end). Value: (extractor names, merged NER labels, clean_cs, clean_ce).
    ``ner_label`` is ORG/LOC/MISC/PRODUCT from spaCy or HF; None if unknown for that path.
    """
    bucket: dict[tuple[int, int], tuple[set[str], set[str], int, int]] = {}
    for o0, o1, name, cs, ce, ner_label in spans:
        if o1 <= o0:
            continue
        key = (o0, o1)
        if key not in bucket:
            lbs: set[str] = set()
            if ner_label:
                lbs.add(ner_label)
            bucket[key] = ({name}, lbs, cs, ce)
        else:
            ex, lbs, ccs, cce = bucket[key]
            ex = set(ex)
            ex.add(name)
            nl = set(lbs)
            if ner_label:
                nl.add(ner_label)
            bucket[key] = (ex, nl, ccs, cce)
    return bucket


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _validate_ner_text_column(windows: list[dict[str, str]], col: str) -> str | None:
    if not windows or not col:
        return None
    if col not in windows[0]:
        keys = list(windows[0].keys())
        preview = ", ".join(keys[:24])
        more = f" (+{len(keys) - 24} more)" if len(keys) > 24 else ""
        return f"CSV has no column {col!r}. Columns (sample): {preview}{more}"
    return None


def resolve_ner_text_column(fieldnames: list[str], requested: str) -> str:
    """``auto`` → ``org_mentions_union_filtered`` if present, else ``window_text``."""
    if requested != "auto":
        return requested
    if "org_mentions_union_filtered" in fieldnames:
        return "org_mentions_union_filtered"
    return "window_text"


def write_mentions(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = OUTPUT_FIELDS_RAW
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate_mentions_by_window(
    windows: list[dict[str, str]],
    raw_mention_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggregate raw mention rows by window to produce one row per window with
    aggregated ORG mention list.

        Returns one row per input window with added columns:
            org_mentions_count: count of unique mentions
            org_mentions: semicolon-separated list of unique mentions
            mention_types: semicolon-separated list of mention types (aligned with org_mentions)
    """
    # Build window -> unique mentions mapping
    mentions_by_window: dict[str, list[str]] = {}
    mention_types_by_window: dict[str, list[str]] = {}
    entity_attrs_by_window: dict[str, list[str]] = {}
    seen_by_window: dict[str, set[str]] = {}

    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        if w_id:
            mentions_by_window[w_id] = []
            mention_types_by_window[w_id] = []
            entity_attrs_by_window[w_id] = []
            seen_by_window[w_id] = set()

    ner_input_by_wid: dict[str, str] = {}
    for mention_row in raw_mention_rows:
        w_id = (mention_row.get("window_id") or "").strip()
        m_text = (
            mention_row.get("org_mention_text")
            or mention_row.get("raw_mention")
            or ""
        ).strip()
        if not w_id or not m_text or w_id not in mentions_by_window:
            continue
        m_key = m_text.lower()
        if m_key in seen_by_window[w_id]:
            continue
        seen_by_window[w_id].add(m_key)
        mentions_by_window[w_id].append(m_text)
        mention_type = str(mention_row.get("mention_type") or "ORGANIZATION  ").strip()
        mention_types_by_window[w_id].append(mention_type)
        ent_attr = str(mention_row.get("entity_attribute") or "ORG").strip() or "ORG"
        entity_attrs_by_window[w_id].append(ent_attr)
        nit = str(mention_row.get("ner_input_text") or "").strip()
        if nit and w_id not in ner_input_by_wid:
            ner_input_by_wid[w_id] = nit

    # Prepare output: one row per window with aggregated mentions
    out_rows: list[dict[str, Any]] = []
    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        row_out = dict(w_dict)
        row_out["org_mentions_count"] = str(len(mentions_by_window.get(w_id, [])))
        row_out["org_mention_count"] = row_out["org_mentions_count"]
        row_out["org_mentions"] = " ; ".join(mentions_by_window.get(w_id, []))
        row_out["mention_types"] = " ; ".join(mention_types_by_window.get(w_id, []))
        row_out["entity_attributes"] = " ; ".join(entity_attrs_by_window.get(w_id, []))
        row_out["ner_input_text"] = ner_input_by_wid.get(w_id, "")
        out_rows.append(row_out)

    return out_rows


def build_org_diff_log_rows(
    windows: list[dict[str, str]],
    mention_rows: list[dict[str, Any]],
    nonraw_by_window_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compare span-level ORG harvest (all rows in mention_rows per window) to the nonraw rule
    (first occurrence per case-insensitive mention string).

    - Surplus spans (2nd+ occurrence of the same text) are logged with extra_source=raw and
      diff_reason=duplicate_in_raw.
    - If the aggregated nonraw string list ever contains a text not seen in any span row
      (should not happen with a consistent pipeline), log one row per such text with
      extra_source=nonraw and diff_reason=only_in_nonraw_aggregate.

    Windows with no differences produce no rows.
    """
    by_wid: dict[str, list[dict[str, Any]]] = {}
    for r in mention_rows:
        wid = (r.get("window_id") or "").strip()
        if not wid:
            continue
        by_wid.setdefault(wid, []).append(r)

    out: list[dict[str, Any]] = []
    for w_dict in windows:
        wid = (w_dict.get("window_id") or "").strip()
        mrows = by_wid.get(wid, [])
        raw_total = len(mrows)
        if raw_total == 0:
            continue

        mrows_sorted = sorted(
            mrows,
            key=lambda x: (
                int(x.get("mention_start") or 0),
                int(x.get("mention_end") or 0),
                str(x.get("org_mention_text") or ""),
            ),
        )

        span_texts_lower: set[str] = set()
        for r in mrows_sorted:
            text = str(r.get("org_mention_text") or r.get("raw_mention") or "").strip()
            if text:
                span_texts_lower.add(text.lower())

        uniq_after = len(span_texts_lower)

        seen_lower: set[str] = set()
        for r in mrows_sorted:
            text = str(r.get("org_mention_text") or r.get("raw_mention") or "").strip()
            low = text.lower()
            if not low:
                continue
            if low in seen_lower:
                out.append({
                    "window_id": wid,
                    "raw_org_span_count": raw_total,
                    "nonraw_unique_org_count": uniq_after,
                    "raw_mention": text,
                    "mention_start": r.get("mention_start", ""),
                    "mention_end": r.get("mention_end", ""),
                    "extractor_name": r.get("extractor_name", ""),
                    "extra_source": "raw",
                    "diff_reason": "duplicate_in_raw",
                })
            else:
                seen_lower.add(low)

        # Defensive: strings appearing in *_nonraw aggregate but not in any span row
        nonraw_row = nonraw_by_window_id.get(wid)
        if nonraw_row:
            agg_parts = [
                p.strip()
                for p in str(nonraw_row.get("org_mentions") or "").split(" ; ")
                if p.strip()
            ]
            for phrase in agg_parts:
                low = phrase.lower()
                if low and low not in span_texts_lower:
                    out.append({
                        "window_id": wid,
                        "raw_org_span_count": raw_total,
                        "nonraw_unique_org_count": uniq_after,
                        "raw_mention": phrase,
                        "mention_start": "",
                        "mention_end": "",
                        "extractor_name": "",
                        "extra_source": "nonraw",
                        "diff_reason": "only_in_nonraw_aggregate",
                    })

    return out


def aggregate_raw_mentions_by_window(
    windows: list[dict[str, str]],
    raw_mention_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One-row-per-window raw output (no dedupe): preserve mention sequence and details."""
    mentions_by_window: dict[str, list[dict[str, str]]] = {}
    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        if w_id:
            mentions_by_window[w_id] = []

    for mention_row in raw_mention_rows:
        w_id = (mention_row.get("window_id") or "").strip()
        if not w_id or w_id not in mentions_by_window:
            continue
        mentions_by_window[w_id].append({
            "text": str(mention_row.get("org_mention_text") or mention_row.get("raw_mention") or "").strip(),
            "start": str(mention_row.get("mention_start") or ""),
            "end": str(mention_row.get("mention_end") or ""),
            "extractor": str(mention_row.get("extractor_name") or ""),
            "ner_input": str(mention_row.get("ner_input_text") or ""),
            "mention_type": str(mention_row.get("mention_type") or "ORGANIZATION  "),
            "entity_attribute": str(mention_row.get("entity_attribute") or "ORG"),
        })

    out_rows: list[dict[str, Any]] = []
    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        items = mentions_by_window.get(w_id, [])
        ner_input_text = ""
        for it in items:
            if it.get("ner_input"):
                ner_input_text = str(it["ner_input"])
                break
        row_out = {
            "window_id": w_dict.get("window_id", ""),
            "source_company": w_dict.get("source_company", ""),
            "source_cik": w_dict.get("source_cik", ""),
            "filing_year": w_dict.get("filing_year", ""),
            "section": w_dict.get("section", ""),
            "cue_phrase": w_dict.get("cue_phrase", ""),
            "cue_group": w_dict.get("cue_group", ""),
            "trigger_sentence": w_dict.get("trigger_sentence", ""),
            "window_text": w_dict.get("window_text", ""),
            "ner_input_text": ner_input_text,
            "org_mentions_raw_count": str(len(items)),
            "org_mentions_raw": " ; ".join(i["text"] for i in items if i["text"]),
            "mention_types_raw": " ; ".join(i["mention_type"] for i in items if i["mention_type"]),
            "entity_attributes_raw": " ; ".join(i["entity_attribute"] for i in items if i.get("entity_attribute")),
            "mention_starts_raw": " ; ".join(i["start"] for i in items if i["start"]),
            "mention_ends_raw": " ; ".join(i["end"] for i in items if i["end"]),
        }
        out_rows.append(row_out)

    return out_rows


def print_preview(title: str, path: Path, n: int = 5) -> None:
    print(f"\n--- {title} ({path.name}) — first {n} data rows ---")
    if not path.is_file():
        print(f"(file missing: {path})")
        return
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        print(" | ".join(fieldnames))
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(" | ".join(str(row.get(h, ""))[:80] for h in fieldnames))


def process_windows(
    windows: list[dict[str, str]],
    nlp: Any | None,
    ner_pipe: Any,
    type_classifier: Any | None,
    hf_extractor_tag: str = DEFAULT_HF_EXTRACTOR_TAG,
    ner_text_column: str = "window_text",
    hf_keep_labels: frozenset[str] = DEFAULT_HF_NER_LABELS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ner_signal_cache: dict[str, tuple[bool, str | None]] = {}
    type_cache: dict[str, str] = {}
    iterator = tqdm(windows, desc="Processing windows for NER") if tqdm else windows
    for wrow in iterator:
        window_id = (wrow.get("window_id") or "").strip()
        window_full = wrow.get("window_text") or ""
        ner_orig = (wrow.get(ner_text_column) or "").strip()
        clean, o_starts, o_ends = normalize_with_offset_map(ner_orig)
        orig_len = len(ner_orig)

        if not clean.strip():
            continue

        doc = nlp(clean) if nlp is not None else None
        sent_bounds: list[tuple[int, int]] | None = None
        sent_texts: list[str] | None = None
        if doc is not None:
            sents = list(doc.sents)
            sent_bounds = [(s.start_char, s.end_char) for s in sents]
            sent_texts = [s.text for s in sents]

        span_extractors: list[tuple[int, int, str, int, int, str | None]] = []

        if doc is not None:
            for cs, ce, slbl in extract_spacy_labeled_spans(doc):
                mapped = clean_span_to_orig(cs, ce, o_starts, o_ends, orig_len)
                if mapped is None:
                    continue
                o0, o1 = mapped
                span_extractors.append((o0, o1, "spacy", cs, ce, slbl))

        for cs, ce, hlbl in extract_hf_labeled_spans(ner_pipe, clean, hf_keep_labels):
            mapped = clean_span_to_orig(cs, ce, o_starts, o_ends, orig_len)
            if mapped is None:
                continue
            o0, o1 = mapped
            span_extractors.append((o0, o1, hf_extractor_tag, cs, ce, hlbl))

        merged = merge_spans_for_window(span_extractors)

        # Stable order: by start, then end, then extractor label string
        for (o0, o1) in sorted(merged.keys(), key=lambda t: (t[0], t[1])):
            ex_names, ner_lab_set, cs_ent, ce_ent = merged[(o0, o1)]
            entity_attribute = resolve_entity_attribute(ner_lab_set)
            raw_mention = ner_orig[o0:o1]
            sentence_text = (
                sentence_for_span_orig(
                    ner_orig,
                    doc,
                    cs_ent,
                    ce_ent,
                    o_starts,
                    o_ends,
                    sent_bounds=sent_bounds,
                    sent_texts=sent_texts,
                )
                if doc is not None
                else ""
            )

            rm = raw_mention.strip()
            if TICKER_PATTERN.match(rm):
                mention_type = "TICKER"
            elif entity_attribute == "LOC":
                mention_type = "REGION"
            elif entity_attribute == "PRODUCT":
                mention_type = "ComputingProduct"
            else:
                mention_type = classify_mention_type(
                    raw_mention,
                    window_full,
                    nlp=nlp,
                    ner_pipe=ner_pipe,
                    type_classifier=type_classifier,
                    ner_signal_cache=ner_signal_cache,
                    type_cache=type_cache,
                )
            rows.append({
                "window_id": window_id,
                "source_company": wrow.get("source_company", ""),
                "source_cik": wrow.get("source_cik", ""),
                "filing_year": wrow.get("filing_year", ""),
                "section": wrow.get("section", ""),
                "cue_phrase": wrow.get("cue_phrase", ""),
                "cue_group": wrow.get("cue_group", ""),
                "trigger_sentence": wrow.get("trigger_sentence", ""),
                "window_text": window_full,
                "ner_input_text": ner_orig,
                "org_mention_text": raw_mention,
                "mention_start": o0,
                "mention_end": o1,
                "mention_type": mention_type,
                "entity_attribute": entity_attribute,
                "sentence_text": sentence_text,
                "extractor_name": "+".join(sorted(ex_names)),
            })

    rows.sort(key=lambda r: (r["window_id"], r["mention_start"], r["mention_end"], r["extractor_name"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest raw ORG mentions from explicit candidate windows.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("explicit_candidate_windows.csv"),
        help="Input CSV from build_explicit_candidate_windows.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("explicit_mentions_raw.csv"),
        help="Output CSV (one row per raw mention span)",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model name (larger models often improve recall)",
    )
    parser.add_argument(
        "--disable-spacy",
        action="store_true",
        help="Skip spaCy and run HF NER only (useful when spaCy is unavailable).",
    )
    parser.add_argument(
        "--device",
        default="-1",
        help="HF pipeline device: -1 CPU, 0+ GPU index",
    )
    parser.add_argument(
        "--hf-ner-model",
        default=DEFAULT_HF_NER_MODEL,
        help="Hugging Face model id for token-classification NER (default: Jean-Baptiste/roberta-large-ner-english)",
    )
    parser.add_argument(
        "--hf-keep-labels",
        default=DEFAULT_HF_NER_LABELS_STR,
        metavar="LABELS",
        help=(
            "Comma-separated HF NER labels to keep as spans (default: "
            f"{DEFAULT_HF_NER_LABELS_STR}). CoNLL models use ORG,LOC,MISC,PER; "
            "add PRODUCT only if your checkpoint predicts it (fine-tuned)."
        ),
    )
    parser.add_argument(
        "--hf-type-model",
        default=DEFAULT_HF_TYPE_MODEL,
        help="HF model id for mention-type classification (default: facebook/bart-large-mnli)",
    )
    parser.add_argument(
        "--disable-type-classifier",
        action="store_true",
        help="Disable model-driven mention type classification; falls back to ORGANIZATION   when non-ticker/non-region.",
    )
    parser.add_argument(
        "--hf-extractor-tag",
        default=DEFAULT_HF_EXTRACTOR_TAG,
        help="Label stored in extractor_name for HF spans (e.g. roberta_ner, distilbert_ner)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Print this many sample rows from input and output CSV (0 to skip)",
    )
    parser.add_argument(
        "--org-diff-log",
        type=Path,
        default=None,
        help="CSV for ORG mention differences (raw vs nonraw). Default: <output stem>_org_diff.csv beside -o",
    )
    parser.add_argument(
        "--ner-text-column",
        default="auto",
        metavar="COL",
        help=(
            "CSV column passed to NER: auto (org_mentions_union_filtered if present, else window_text), "
            "or an explicit column name. Offsets are into ner_input_text; window_text is preserved."
        ),
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    outp = args.output.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    try:
        dev: int | str = int(args.device)
    except ValueError:
        dev = -1

    nlp = None
    if not args.disable_spacy:
        try:
            print("Loading spaCy…")
            nlp = load_spacy(args.spacy_model)
        except Exception as e:
            print(f"Warning: spaCy unavailable ({e}). Continuing with HF NER only.")
            nlp = None
    else:
        print("Skipping spaCy (--disable-spacy).")
    print(f"Loading Hugging Face NER ({args.hf_ner_model})…")
    ner_pipe = load_hf_ner(model_name=args.hf_ner_model, device=dev)

    type_classifier = None
    if args.disable_type_classifier:
        print("Skipping type classifier (--disable-type-classifier).")
    else:
        print(f"Loading HF mention-type classifier ({args.hf_type_model})…")
        type_classifier = load_hf_type_classifier(model_name=args.hf_type_model, device=dev)

    windows = read_windows(inp)
    fieldnames = list(windows[0].keys()) if windows else []
    ner_col = resolve_ner_text_column(fieldnames, args.ner_text_column)
    col_err = _validate_ner_text_column(windows, ner_col)
    if col_err:
        print(col_err, file=sys.stderr)
        return 1
    print(
        f"Processing {len(windows)} windows from {inp} "
        f"(NER column={ner_col!r}, requested={args.ner_text_column!r})"
    )
    hf_keep = parse_hf_ner_labels(args.hf_keep_labels)
    if not hf_keep:
        print("Warning: --hf-keep-labels is empty; HF will contribute no spans.", file=sys.stderr)
    mention_rows = process_windows(
        windows,
        nlp,
        ner_pipe,
        type_classifier,
        hf_extractor_tag=args.hf_extractor_tag,
        ner_text_column=ner_col,
        hf_keep_labels=hf_keep,
    )
    raw_rows = aggregate_raw_mentions_by_window(windows, mention_rows)
    write_mentions(outp, raw_rows, fieldnames=OUTPUT_FIELDS_RAW)
    print(f"Wrote {len(raw_rows)} raw window rows to {outp}")

    # Generate nonraw window-level output with aggregated mentions
    nonraw_outp = outp.parent / (outp.stem + "_nonraw" + outp.suffix)
    nonraw_rows = aggregate_mentions_by_window(windows, mention_rows)
    write_mentions(nonraw_outp, nonraw_rows, fieldnames=OUTPUT_FIELDS_NONRAW)
    print(f"Wrote {len(nonraw_rows)} window rows to {nonraw_outp}")

    nonraw_by_id = {(r.get("window_id") or "").strip(): r for r in nonraw_rows}
    diff_path = (
        args.org_diff_log.resolve()
        if args.org_diff_log is not None
        else (outp.parent / (outp.stem + "_org_diff" + outp.suffix))
    )
    diff_rows = build_org_diff_log_rows(windows, mention_rows, nonraw_by_id)
    write_mentions(diff_path, diff_rows, fieldnames=OUTPUT_FIELDS_ORG_DIFF)
    print(f"Wrote {len(diff_rows)} org-diff rows to {diff_path}")

    if args.preview > 0:
        print_preview("INPUT", inp, args.preview)
        print_preview("OUTPUT (raw)", outp, args.preview)
        print_preview("OUTPUT (nonraw)", nonraw_outp, args.preview)
        print_preview("OUTPUT (org diff)", diff_path, args.preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())