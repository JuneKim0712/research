#!/usr/bin/env python3
"""
Remove funds and trusts from existing company_nodes.csv files (in-place).

Usage:
    python cleanup_company_nodes.py --year 2024
    python cleanup_company_nodes.py --year 2023
    python cleanup_company_nodes.py --year 2024 2023  # clean both years
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Regex patterns to exclude funds, trusts, and NASDAQ entities
_EXCLUDE_COMPANY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\btrustee\b"),
    re.compile(r"(?i)\btrust(?:\s+of\s+)?(?:account|fund|estate)?\b"),
    re.compile(r"(?i)\b(?:mutual|index|exchange-traded|closed-end|open-end|money market|hedge|pension|private)?(\s+)fund\b"),
    re.compile(r"(?i)\bfund\s+of\s+funds\b"),
    re.compile(r"(?i)nasdaq(?:\s*-?\s*(?:100|composite|global|capital market))?(?:\s+index)?$"),
    re.compile(r"(?i)^nasdaq\b"),
    re.compile(r"(?i)\bindex\s+fund\b"),
    re.compile(r"(?i)\bclosed[\s-]?end\b"),
    re.compile(r"(?i)\blagged\s+trust\b"),
)


def _should_exclude_company(company_name: str) -> bool:
    """Return True if the company should be excluded."""
    if not company_name:
        return False
    
    for pattern in _EXCLUDE_COMPANY_PATTERNS:
        if pattern.search(company_name):
            return True
    
    return False


def cleanup_csv(csv_path: Path) -> None:
    """Remove funds and trusts from CSV file, write back to same file."""
    if not csv_path.is_file():
        print(f"[skip] {csv_path} not found")
        return
    
    print(f"\nProcessing: {csv_path}")
    
    # Read all rows
    rows_in: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)
    
    if not rows_in:
        print(f"  [empty] No rows to process")
        return
    
    # Filter out excluded companies
    rows_out: list[dict[str, str]] = []
    excluded: list[str] = []
    
    iterator = tqdm(rows_in, desc="Filtering", leave=False) if tqdm else rows_in
    for row in iterator:
        company_name = row.get("company_name", "").strip()
        if _should_exclude_company(company_name):
            excluded.append(company_name)
        else:
            rows_out.append(row)
    
    # Write back to same file
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["company_name", "cik", "ticker", "gics_sector"])
        writer.writeheader()
        writer.writerows(rows_out)
    
    # Write excluded companies to audit file
    excluded_path = csv_path.with_stem(csv_path.stem + ".excluded")
    if excluded:
        with open(excluded_path, "w", encoding="utf-8") as f:
            f.write(f"Excluded companies (funds/trusts): {len(excluded)}\n\n")
            for name in sorted(set(excluded)):
                f.write(name + "\n")
    
    print(f"  ✓ Input rows:    {len(rows_in):,}")
    print(f"  ✓ Output rows:   {len(rows_out):,}")
    print(f"  ✓ Excluded:      {len(excluded):,}")
    print(f"  ✓ Saved to:      {csv_path}")
    if excluded:
        print(f"  ✓ Audit log:     {excluded_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove funds and trusts from company_nodes.csv files (in-place)"
    )
    parser.add_argument(
        "years",
        nargs="+",
        help="Year(s) to clean (e.g., 2023 2024)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing year folders (default: current dir)"
    )
    
    args = parser.parse_args()
    base = args.base_dir.resolve()
    
    for year_str in args.years:
        year = year_str.strip()
        csv_path = base / year / f"{year}_company_nodes.csv"
        cleanup_csv(csv_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
