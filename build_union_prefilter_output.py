#!/usr/bin/env python3
"""
Build ORG prefilter output CSV from mention-list windows.

Typical use:
  python build_union_prefilter_output.py \
    -i explicit_mentions_raw_nonraw.csv \
    -o ORG_detection/union_filtered_prefilter_output.csv

This script applies org_mention_prefilter.label_mention() to each mention in a
semicolon-separated mention column and writes:
  - org_mentions_union_count
  - org_mentions_union
  - org_mentions_union_filtered
  - prefilter_* audit columns

It preserves all original input columns and appends missing output columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from org_mention_prefilter import label_mention

DEFAULT_MENTION_CANDIDATES = (
    "org_mentions_union",
    "org_mentions_union_raw",
    "org_mentions",
    "org_mentions_raw",
)

DEFAULT_COUNT_CANDIDATES = (
    "org_mentions_union_count",
    "org_mention_count_union",
    "org_mentions_count",
    "org_mentions_raw_count",
)

DEFAULT_MENTION_TYPE_CANDIDATES = (
    "mention_types",
    "mention_types_raw",
)

OUTPUT_APPEND_COLUMNS = [
    "org_mentions_union_count",
    "org_mentions_union",
    "mention_types_union",
    "org_mentions_union_filtered",
    "mention_types_union_filtered",
    "prefilter_input_column",
    "prefilter_total_mentions",
    "prefilter_keep_count",
    "prefilter_drop_count",
    "prefilter_review_count",
    "prefilter_keep_mentions",
    "prefilter_drop_mentions",
    "prefilter_review_mentions",
    "prefilter_keep_reasons",
    "prefilter_drop_reasons",
    "prefilter_review_reasons",
    "prefilter_keep_count_by_type",
    "prefilter_drop_count_by_type",
    "prefilter_review_count_by_type",
]


def pick_existing(fieldnames: list[str], candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def split_mentions(raw: str, sep: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [part.strip() for part in raw.split(sep) if part.strip()]


def join_mentions(values: list[str], sep: str) -> str:
    return sep.join(values)


def safe_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def resolve_mention_column(fieldnames: list[str], requested: str) -> str:
    if requested != "auto":
        if requested not in fieldnames:
            raise SystemExit(
                f"Mention column {requested!r} not found. Available columns: {fieldnames}"
            )
        return requested

    found = pick_existing(fieldnames, DEFAULT_MENTION_CANDIDATES)
    if found is None:
        raise SystemExit(
            "Could not auto-detect mention column. Expected one of: "
            f"{', '.join(DEFAULT_MENTION_CANDIDATES)}"
        )
    return found


def count_by_type(mentions: list[str], types: list[str], sep: str = " ; ") -> str:
    """
    Count mentions by type and return formatted string like "COMPANY:5 ; PRODUCT:2 ; SERVICE:1".
    """
    from collections import Counter
    
    type_counter: Counter[str] = Counter()
    for m_type in types:
        if m_type.strip():
            type_counter[m_type.strip()] += 1
    
    if not type_counter:
        return ""
    
    # Sort by type name for consistent output
    sorted_counts = sorted(type_counter.items())
    result = sep.join(f"{t}:{c}" for t, c in sorted_counts)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build union_filtered_prefilter_output.csv using org_mention_prefilter rules"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("explicit_mentions_raw_nonraw.csv"),
        help="Input CSV with semicolon-separated mention list",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ORG_detection/union_filtered_prefilter_output.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--mentions-column",
        default="auto",
        help=(
            "Mention list column to prefilter. Use 'auto' to detect from: "
            "org_mentions_union, org_mentions_union_raw, org_mentions, org_mentions_raw"
        ),
    )
    parser.add_argument(
        "--source-column",
        default="source_company",
        help="Optional source company column for self-name suppression",
    )
    parser.add_argument(
        "--sep",
        default=" ; ",
        help="Mention separator used in mention-list columns",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    with input_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        mention_col = resolve_mention_column(fieldnames, args.mentions_column)
        count_col = pick_existing(fieldnames, DEFAULT_COUNT_CANDIDATES)
        mention_type_col = pick_existing(fieldnames, DEFAULT_MENTION_TYPE_CANDIDATES)
        has_source = args.source_column in fieldnames

        # Convert reader to list to support tqdm
        rows_in = list(reader)
        rows_out: list[dict[str, str]] = []
        totals = {
            "rows": 0,
            "mentions": 0,
            "keep": 0,
            "drop": 0,
            "review": 0,
        }

        iterator = tqdm(rows_in, desc="Prefiltering mentions") if tqdm else rows_in
        for row in iterator:
            raw_mentions = split_mentions(row.get(mention_col, "") or "", args.sep)
            raw_mention_types = []
            if mention_type_col:
                raw_mention_types = split_mentions(row.get(mention_type_col, "") or "", args.sep)
            # Ensure types list is same length as mentions (pad with "COMPANY" if needed)
            while len(raw_mention_types) < len(raw_mentions):
                raw_mention_types.append("COMPANY")
            # Trim if types exceed mentions (shouldn't happen)
            raw_mention_types = raw_mention_types[:len(raw_mentions)]
            
            source_company = (row.get(args.source_column) or "").strip() if has_source else ""

            keep_mentions: list[str] = []
            keep_mention_types: list[str] = []
            drop_mentions: list[str] = []
            drop_mention_types: list[str] = []
            review_mentions: list[str] = []
            review_mention_types: list[str] = []

            keep_reasons: list[str] = []
            drop_reasons: list[str] = []
            review_reasons: list[str] = []

            for mention, mention_type in zip(raw_mentions, raw_mention_types):
                label, reason = label_mention(mention, source_company=source_company)
                if label == "keep":
                    keep_mentions.append(mention)
                    keep_mention_types.append(mention_type)
                    keep_reasons.append(reason)
                elif label == "drop_obvious_junk":
                    drop_mentions.append(mention)
                    drop_mention_types.append(mention_type)
                    drop_reasons.append(reason)
                else:
                    review_mentions.append(mention)
                    review_mention_types.append(mention_type)
                    review_reasons.append(reason)

            # Preserve explicit union count if available and valid; else compute from mention list.
            union_count_val: int
            if count_col:
                parsed = safe_int(row.get(count_col, ""))
                union_count_val = parsed if parsed is not None else len(raw_mentions)
            else:
                union_count_val = len(raw_mentions)

            out = dict(row)
            out["org_mentions_union_count"] = str(union_count_val)
            out["org_mentions_union"] = join_mentions(raw_mentions, args.sep)
            out["mention_types_union"] = join_mentions(raw_mention_types, args.sep)
            out["org_mentions_union_filtered"] = join_mentions(keep_mentions, args.sep)
            out["mention_types_union_filtered"] = join_mentions(keep_mention_types, args.sep)

            out["prefilter_input_column"] = mention_col
            out["prefilter_total_mentions"] = str(len(raw_mentions))
            out["prefilter_keep_count"] = str(len(keep_mentions))
            out["prefilter_drop_count"] = str(len(drop_mentions))
            out["prefilter_review_count"] = str(len(review_mentions))
            out["prefilter_keep_mentions"] = join_mentions(keep_mentions, args.sep)
            out["prefilter_drop_mentions"] = join_mentions(drop_mentions, args.sep)
            out["prefilter_review_mentions"] = join_mentions(review_mentions, args.sep)
            out["prefilter_keep_reasons"] = join_mentions(keep_reasons, args.sep)
            out["prefilter_drop_reasons"] = join_mentions(drop_reasons, args.sep)
            out["prefilter_review_reasons"] = join_mentions(review_reasons, args.sep)
            
            # Track counts by mention type for transparency in drop calculation
            out["prefilter_keep_count_by_type"] = count_by_type(keep_mentions, keep_mention_types, args.sep)
            out["prefilter_drop_count_by_type"] = count_by_type(drop_mentions, drop_mention_types, args.sep)
            out["prefilter_review_count_by_type"] = count_by_type(review_mentions, review_mention_types, args.sep)

            rows_out.append(out)

            totals["rows"] += 1
            totals["mentions"] += len(raw_mentions)
            totals["keep"] += len(keep_mentions)
            totals["drop"] += len(drop_mentions)
            totals["review"] += len(review_mentions)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_fields = list(fieldnames)
    for col in OUTPUT_APPEND_COLUMNS:
        if col not in output_fields:
            output_fields.append(col)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Prefilter mention column: {mention_col}")
    print(f"Rows processed: {totals['rows']}")
    print(f"Mentions total: {totals['mentions']}")
    print(
        "Mentions kept/dropped/review: "
        f"{totals['keep']}/{totals['drop']}/{totals['review']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
