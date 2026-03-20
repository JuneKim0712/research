#!/usr/bin/env python3
"""
Raw organization / company mention harvest from explicit_candidate_windows.csv.

Runs two off-the-shelf extractors on each window_text (recall-first union):
  1) spaCy English NER — keep spans labeled ORG (optionally GPE for some holdings; off by default).
  2) Hugging Face token-classification NER (default: Jean-Baptiste/roberta-large-ner-english, RoBERTa
     on CoNLL-2003) — keep ORG and MISC (MISC often catches company-like phrases).

Same (window_id, char span in *original* window_text) from both models is merged into one row
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
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

# --- Optional heavy imports (clear error if missing) -------------------------------------------
try:
    import spacy
except ImportError as e:  # pragma: no cover
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

# Default HF model: RoBERTa-large CoNLL English (ORG / LOC / PER / MISC); not BERT.
DEFAULT_HF_NER_MODEL = "Jean-Baptiste/roberta-large-ner-english"
DEFAULT_HF_EXTRACTOR_TAG = "roberta_ner"

# Zero-width / format characters to drop (length changes — handled in mapping pass).
_ZERO_WIDTH = frozenset(
    "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\ufeff\u00ad"
)

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
    "org_mention_count",
    "org_mentions_raw",
    "mention_starts_raw",
    "mention_ends_raw",
    "extractors_raw",
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
    "org_mention_count",
    "org_mentions",
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


def sentence_for_span_orig(
    window_orig: str,
    doc: Any,
    cs: int,
    ce: int,
    orig_starts: list[int],
    orig_ends_excl: list[int],
) -> str:
    """Original-window substring for the spaCy sentence containing the mention (clean offsets)."""
    orig_len = len(window_orig)
    mid = (cs + ce) // 2
    for sent in doc.sents:
        if sent.start_char <= mid < sent.end_char:
            s0, s1 = sent.start_char, sent.end_char
            if s0 >= len(orig_starts):
                return sent.text
            s1_adj = min(s1, len(orig_starts)) - 1
            if s1_adj < s0:
                return sent.text
            o0 = orig_starts[s0]
            o1 = orig_ends_excl[s1_adj]
            if 0 <= o0 <= o1 <= orig_len:
                return window_orig[o0:o1]
            return sent.text
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


def extract_spacy_org(doc: Any) -> list[tuple[int, int]]:
    """ORG spans in *clean* text character offsets (reuse parsed doc)."""
    out: list[tuple[int, int]] = []
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        if ent.start_char is None or ent.end_char is None:
            continue
        out.append((ent.start_char, ent.end_char))
    return out


def extract_hf_org_misc(ner_pipe: Any, clean_text: str) -> list[tuple[int, int, str]]:
    if not clean_text.strip():
        return []
    # pipeline returns dicts with entity_group, start, end (char indices into input string)
    raw = ner_pipe(clean_text)
    out: list[tuple[int, int, str]] = []
    for item in raw:
        label = item.get("entity_group") or item.get("entity") or ""
        label = str(label).upper().replace("B-", "").replace("I-", "")
        if label not in ("ORG", "MISC"):
            continue
        start = int(item["start"])
        end = int(item["end"])
        if end > start:
            out.append((start, end, label))
    return out


def merge_spans_for_window(
    spans: Iterable[tuple[int, int, str, int, int]],
) -> dict[tuple[int, int], tuple[set[str], int, int]]:
    """
    Key: (orig_start, orig_end). Value: (extractor names, clean_cs, clean_ce) for sentence lookup.
    When the same span is emitted twice with identical mapping, clean coords should match.
    """
    bucket: dict[tuple[int, int], tuple[set[str], int, int]] = {}
    for o0, o1, name, cs, ce in spans:
        if o1 <= o0:
            continue
        key = (o0, o1)
        if key not in bucket:
            bucket[key] = ({name}, cs, ce)
        else:
            ex, ccs, cce = bucket[key]
            ex.add(name)
            bucket[key] = (ex, ccs, cce)
    return bucket


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
            org_mention_count: count of unique mentions
            org_mentions: semicolon-separated list of unique mentions
    """
    # Build window -> unique mentions mapping
    mentions_by_window: dict[str, list[str]] = {}
    seen_by_window: dict[str, set[str]] = {}

    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        if w_id:
            mentions_by_window[w_id] = []
            seen_by_window[w_id] = set()

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

    # Prepare output: one row per window with aggregated mentions
    out_rows: list[dict[str, Any]] = []
    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        row_out = dict(w_dict)
        row_out["org_mention_count"] = str(len(mentions_by_window.get(w_id, [])))
        row_out["org_mentions"] = " ; ".join(mentions_by_window.get(w_id, []))
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
        })

    out_rows: list[dict[str, Any]] = []
    for w_dict in windows:
        w_id = (w_dict.get("window_id") or "").strip()
        items = mentions_by_window.get(w_id, [])
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
            "org_mention_count": str(len(items)),
            "org_mentions_raw": " ; ".join(i["text"] for i in items if i["text"]),
            "mention_starts_raw": " ; ".join(i["start"] for i in items if i["start"]),
            "mention_ends_raw": " ; ".join(i["end"] for i in items if i["end"]),
            "extractors_raw": " ; ".join(i["extractor"] for i in items if i["extractor"]),
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
    nlp: Any,
    ner_pipe: Any,
    hf_extractor_tag: str = DEFAULT_HF_EXTRACTOR_TAG,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wrow in windows:
        window_id = (wrow.get("window_id") or "").strip()
        window_orig = wrow.get("window_text") or ""
        clean, o_starts, o_ends = normalize_with_offset_map(window_orig)
        orig_len = len(window_orig)

        if not clean.strip():
            continue

        doc = nlp(clean)

        span_extractors: list[tuple[int, int, str, int, int]] = []

        for cs, ce in extract_spacy_org(doc):
            mapped = clean_span_to_orig(cs, ce, o_starts, o_ends, orig_len)
            if mapped is None:
                continue
            o0, o1 = mapped
            span_extractors.append((o0, o1, "spacy", cs, ce))

        for cs, ce, _lbl in extract_hf_org_misc(ner_pipe, clean):
            mapped = clean_span_to_orig(cs, ce, o_starts, o_ends, orig_len)
            if mapped is None:
                continue
            o0, o1 = mapped
            span_extractors.append((o0, o1, hf_extractor_tag, cs, ce))

        merged = merge_spans_for_window(span_extractors)

        # Stable order: by start, then end, then extractor label string
        for (o0, o1) in sorted(merged.keys(), key=lambda t: (t[0], t[1])):
            ex_names, cs_ent, ce_ent = merged[(o0, o1)]
            raw_mention = window_orig[o0:o1]
            sentence_text = sentence_for_span_orig(
                window_orig, doc, cs_ent, ce_ent, o_starts, o_ends
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
                "window_text": window_orig,
                "org_mention_text": raw_mention,
                "mention_start": o0,
                "mention_end": o1,
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
        "--device",
        default="-1",
        help="HF pipeline device: -1 CPU, 0+ GPU index",
    )
    parser.add_argument(
        "--hf-ner-model",
        default=DEFAULT_HF_NER_MODEL,
        help="Hugging Face model id for token-classification NER (default: RoBERTa-large CoNLL English)",
    )
    parser.add_argument(
        "--hf-extractor-tag",
        default=DEFAULT_HF_EXTRACTOR_TAG,
        help="Label stored in extractor_name for HF spans (e.g. roberta_ner, dslim_bert_ner)",
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

    print("Loading spaCy…")
    nlp = load_spacy(args.spacy_model)
    print(f"Loading Hugging Face NER ({args.hf_ner_model})…")
    ner_pipe = load_hf_ner(model_name=args.hf_ner_model, device=dev)

    windows = read_windows(inp)
    print(f"Processing {len(windows)} windows from {inp}")
    mention_rows = process_windows(
        windows, nlp, ner_pipe, hf_extractor_tag=args.hf_extractor_tag
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