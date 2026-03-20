"""Remove prefilter processing columns from union_filtered_prefilter_output.csv."""
import csv
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Canonical output columns (normalized names)
KEEP_COLUMNS = [
    "window_id",
    "source_company",
    "source_cik",
    "filing_year",
    "section",
    "cue_phrase",
    "cue_group",
    "trigger_sentence",
    "window_text",
    "org_mentions_union_count",
    "org_mentions_union",
    "org_mentions_union_filtered",
    "org_mentions_removed_count",
]

INPUT_COLUMN_ALIASES = {
    "org_mentions_union_count": ["org_mentions_union_count", "org_mention_count_union"],
    "org_mentions_union": ["org_mentions_union", "org_mentions_union_raw"],
    "org_mentions_union_filtered": ["org_mentions_union_filtered"],
}

# Columns to remove (all prefilter_* columns)
REMOVE_COLUMNS = {
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
}


def clean_csv(input_path: str, output_path: str = None) -> None:
    """Remove prefilter columns from CSV and write cleaned version."""
    if output_path is None:
        input_p = Path(input_path)
        output_path = input_p.with_stem(input_p.stem + "_cleaned")

    print(f"Reading from: {input_path}")
    rows = []

    def count_mentions(mention_str: str, sep: str = " ; ") -> int:
        """Count semicolon-separated mentions."""
        if not mention_str or not mention_str.strip():
            return 0
        return len([m for m in mention_str.split(sep) if m.strip()])

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        def pick_column(canonical: str) -> str | None:
            aliases = INPUT_COLUMN_ALIASES.get(canonical, [canonical])
            for name in aliases:
                if name in fieldnames:
                    return name
            return None

        input_union_count_col = pick_column("org_mentions_union_count")
        input_union_col = pick_column("org_mentions_union")
        input_filtered_col = pick_column("org_mentions_union_filtered")

        if input_union_col is None or input_filtered_col is None:
            raise SystemExit(
                "Missing required mention columns. "
                f"Need union/filtered columns in header; found: {fieldnames}"
            )

        print(f"  Original columns: {len(fieldnames)}")
        print(f"  Removed columns: {len(REMOVE_COLUMNS & set(fieldnames))}")

        # Convert reader to list to support tqdm
        rows_list = list(reader)
        iterator = tqdm(rows_list, desc="Cleaning columns") if tqdm else rows_list
        for row in iterator:
            union_mentions = count_mentions(row.get(input_union_col, ""))
            filtered_mentions = count_mentions(row.get(input_filtered_col, ""))
            mentions_removed = union_mentions - filtered_mentions

            filtered_row = {
                "window_id": row.get("window_id", ""),
                "source_company": row.get("source_company", ""),
                "source_cik": row.get("source_cik", ""),
                "filing_year": row.get("filing_year", ""),
                "section": row.get("section", ""),
                "cue_phrase": row.get("cue_phrase", ""),
                "cue_group": row.get("cue_group", ""),
                "trigger_sentence": row.get("trigger_sentence", ""),
                "window_text": row.get("window_text", ""),
                "org_mentions_union_count": row.get(input_union_count_col, "")
                if input_union_count_col
                else str(union_mentions),
                "org_mentions_union": row.get(input_union_col, ""),
                "org_mentions_union_filtered": row.get(input_filtered_col, ""),
                "org_mentions_removed_count": str(mentions_removed),
            }

            rows.append(filtered_row)

    # Write cleaned CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Final columns: {len(KEEP_COLUMNS)}")
    print(f"Wrote cleaned file to: {output_path}")
    print(f"  Rows: {len(rows)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove prefilter processing columns from union_filtered_prefilter_output.csv"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="ORG_detection/union_filtered_prefilter_output.csv",
        help="Input CSV file (default: ORG_detection/union_filtered_prefilter_output.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file (default: input_stem_cleaned.csv)",
    )

    args = parser.parse_args()
    clean_csv(args.input, args.output)
