#!/usr/bin/env python3
"""
Reformat a mention / LLM-label CSV into a stable, human-readable column order.

Typical use: after ``llm.py`` (or ``gemini_mention_label.py``), the file keeps the
input column order with ``llm_*`` columns appended. This script:

- Puts **ids → filer → cue → mention span → sentence context → LLM verdicts** first
- Moves **long blobs** (``window_text``, ``ner_input_text``, union mention lists, …) last
- Leaves any unknown columns **alphabetically** in the middle tail (before long blobs)

Examples:
  python format_llm_mention_csv.py -i mentions_labeled.csv -o mentions_labeled_neat.csv
  python format_llm_mention_csv.py -i in.csv -o out.csv --preset minimal --excel-bom
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Short, scan-friendly columns first (only emitted if present in the input).
ORDER_LEADING: tuple[str, ...] = (
    "window_id",
    "source_cik",
    "filing_year",
    "section",
    "source_company",
    "source_company_name",
    "cue_phrase",
    "cue_group",
    "trigger_sentence",
    "mention_org",
    "raw_mention",
    "org_mention_text",
    "mention_start",
    "mention_end",
    "extractor_name",
    "mention_type",
    "sentence_text",
    "source_sentence",
    "llm_label",
    "mention_type_resolved",
    "ticker_match_company",
    "llm_product_type",
    "pipeline_role",
    "llm_confidence",
    "llm_owner_company_candidate",
    "llm_reason",
)

# Wide / noisy columns last (only if present).
ORDER_LONG_TAIL: tuple[str, ...] = (
    "ner_input_text",
    "window_text",
    "org_mentions_union_count",
    "org_mentions_union",
    "mention_types_union",
    "org_mentions_union_filtered",
    "mention_types_union_filtered",
    "org_mentions_removed_count",
    "org_mentions",
    "org_mentions_raw",
    "mention_types",
    "mention_types_raw",
)

PRESET_MINIMAL: frozenset[str] = frozenset(
    {
        "window_id",
        "source_company",
        "source_company_name",
        "source_cik",
        "filing_year",
        "section",
        "cue_phrase",
        "cue_group",
        "trigger_sentence",
        "mention_org",
        "raw_mention",
        "mention_start",
        "mention_end",
        "mention_type",
        "sentence_text",
        "source_sentence",
        "llm_label",
        "mention_type_resolved",
        "ticker_match_company",
        "llm_product_type",
        "pipeline_role",
        "llm_confidence",
        "llm_owner_company_candidate",
        "llm_reason",
    }
)


def _build_fieldnames(all_cols: list[str], *, preset: str) -> list[str]:
    present = set(all_cols)
    if preset == "minimal":
        lead = [c for c in ORDER_LEADING if c in PRESET_MINIMAL and c in present]
        rest = sorted(c for c in PRESET_MINIMAL if c in present and c not in lead)
        return lead + rest

    leading = [c for c in ORDER_LEADING if c in present]
    tail = [c for c in ORDER_LONG_TAIL if c in present]
    used = set(leading) | set(tail)
    rest = [c for c in all_cols if c not in used]
    rest_sorted = sorted(rest)
    return leading + rest_sorted + tail


def main() -> int:
    p = argparse.ArgumentParser(
        description="Reorder CSV columns for readable LLM mention exports.",
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Input CSV")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output CSV")
    p.add_argument(
        "--preset",
        choices=("full", "minimal"),
        default="full",
        help="full = all columns reordered; minimal = compact review sheet",
    )
    p.add_argument(
        "--excel-bom",
        action="store_true",
        help="Write UTF-8 with BOM so Excel recognizes Unicode",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    encoding = "utf-8-sig" if args.excel_bom else "utf-8"
    with inp.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        orig_fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not orig_fields:
        raise SystemExit("Input CSV has no header row")

    out_fields = _build_fieldnames(orig_fields, preset=args.preset)
    if not out_fields:
        raise SystemExit("No columns selected (empty CSV header?)")

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding=encoding, newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} rows, {len(out_fields)} columns -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
