#!/usr/bin/env python3
"""
Remove rows from CSV files when they contain company names listed in exclusion audit logs.

Default logs:
- 2023/2023_company_nodes.excluded.csv
- 2024/2024_company_nodes.excluded.csv

Behavior:
- Scans all CSV files under the workspace root (excluding the audit logs themselves)
- Drops any row where any cell matches an excluded company name exactly, case-insensitive
- Also checks semicolon-separated list cells (" ; ") and comma-separated list cells
- Writes files in-place
- Writes a summary report CSV
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize(s: str) -> str:
    return " ".join((s or "").strip().casefold().split())


def load_exclusions(log_paths: list[Path]) -> set[str]:
    exclusions: set[str] = set()
    for p in log_paths:
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                # Skip header line like: Excluded companies (funds/trusts): 233
                if i == 0 and text.casefold().startswith("excluded companies"):
                    continue
                exclusions.add(normalize(text))
    return exclusions


def cell_matches_excluded(cell: str, excluded: set[str]) -> bool:
    raw = (cell or "").strip()
    if not raw:
        return False

    # Whole-cell exact match
    if normalize(raw) in excluded:
        return True

    # Common list delimiters
    for delim in (" ; ", ";", ","):
        if delim in raw:
            parts = [p.strip() for p in raw.split(delim) if p.strip()]
            for part in parts:
                if normalize(part) in excluded:
                    return True
    return False


def row_matches_excluded(row: dict[str, str], excluded: set[str]) -> bool:
    for value in row.values():
        if cell_matches_excluded(str(value or ""), excluded):
            return True
    return False


def process_csv(path: Path, excluded: set[str]) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return 0, 0

    keep_rows: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        if row_matches_excluded(row, excluded):
            removed += 1
        else:
            keep_rows.append(row)

    if removed > 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(keep_rows)

    return len(rows), removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge excluded companies from all CSV files.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root to scan")
    parser.add_argument(
        "--logs",
        nargs="*",
        type=Path,
        default=[
            Path("2023/2023_company_nodes.excluded.csv"),
            Path("2024/2024_company_nodes.excluded.csv"),
        ],
        help="Exclusion log files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("purge_excluded_companies_report.csv"),
        help="Summary report CSV",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    logs = [(root / p).resolve() if not p.is_absolute() else p.resolve() for p in args.logs]
    report_path = (root / args.report).resolve() if not args.report.is_absolute() else args.report.resolve()

    excluded = load_exclusions(logs)
    if not excluded:
        raise SystemExit("No exclusions loaded from logs.")

    skip_paths = {p.resolve() for p in logs}
    skip_names = {
        "purge_excluded_companies_report.csv",
        "2023_company_nodes.excluded.csv",
        "2024_company_nodes.excluded.csv",
    }

    summaries: list[dict[str, str]] = []

    for csv_path in root.rglob("*.csv"):
        rp = csv_path.resolve()
        if rp in skip_paths:
            continue
        if csv_path.name in skip_names:
            continue

        total_rows, removed_rows = process_csv(rp, excluded)
        if total_rows == 0:
            continue

        if removed_rows > 0:
            rel = rp.relative_to(root)
            summaries.append(
                {
                    "file": rel.as_posix(),
                    "input_rows": str(total_rows),
                    "removed_rows": str(removed_rows),
                    "output_rows": str(total_rows - removed_rows),
                }
            )

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "input_rows", "removed_rows", "output_rows"],
        )
        writer.writeheader()
        writer.writerows(summaries)

    total_removed = sum(int(r["removed_rows"]) for r in summaries)
    print(f"Excluded names loaded: {len(excluded)}")
    print(f"Files changed: {len(summaries)}")
    print(f"Rows removed total: {total_removed}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
