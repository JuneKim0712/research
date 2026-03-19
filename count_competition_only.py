#!/usr/bin/env python3
"""
Count how many files in *_10K_business appear to contain competition-only
sections by scanning the extracted text files directly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COMPETITION_RE = re.compile(r"\b(competition|competitive|competitor|compete)\b", re.IGNORECASE)
BUSINESS_HEADER_RE = re.compile(r"\bitem\s*1\s*[^\n]{0,80}\bbusiness\b", re.IGNORECASE)
COMPETITION_HEADER_RE = re.compile(r"\bitem\s*1\s*[^\n]{0,80}\bcompetition\b", re.IGNORECASE)


def is_competition_only_text(text: str) -> bool:
    """
    Heuristic: classify as competition-only when the extracted text starts like
    a competition section and does not show a business header.
    """
    head = text[:8000]
    has_competition_header = bool(COMPETITION_HEADER_RE.search(head))
    has_business_header = bool(BUSINESS_HEADER_RE.search(head))
    has_competition_terms = bool(COMPETITION_RE.search(head))
    return (has_competition_header and not has_business_header) or (
        has_competition_terms and not has_business_header and "business" not in head.lower()[:300]
    )


def count_competition_only(input_dirs: list[Path]) -> dict:
    """Count competition-only files across one or more *_10K_business folders."""
    existing_dirs = [d for d in input_dirs if d.exists() and d.is_dir()]
    missing_dirs = [str(d) for d in input_dirs if d not in existing_dirs]

    all_files: list[Path] = []
    per_dir_counts: dict[str, int] = {}
    for d in existing_dirs:
        files = sorted(d.glob("*_business.txt"))
        all_files.extend(files)
        per_dir_counts[d.name] = len(files)

    competition_only_files: list[str] = []
    for file_path in all_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if is_competition_only_text(text):
            competition_only_files.append(file_path.name)

    return {
        "input_dirs": [str(d) for d in existing_dirs],
        "missing_dirs": missing_dirs,
        "per_dir_counts": per_dir_counts,
        "competition_only_count": len(competition_only_files),
        "total_business_folder": len(all_files),
        "competition_only_files": sorted(competition_only_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count files in *_10K_business that appear to be competition-only "
            "by scanning the extracted text files directly."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[Path("2023_10K_business"), Path("2024_10K_business")],
        help="One or more *_10K_business folders to scan.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the competition-only filenames",
    )
    args = parser.parse_args()

    result = count_competition_only(args.input_dirs)

    print("\n--- Competition-only files in *_10K_business ---")
    print(f"Input dirs scanned:       {', '.join(result['input_dirs']) if result['input_dirs'] else '(none)'}")
    if result["missing_dirs"]:
        print(f"Missing input dirs:       {', '.join(result['missing_dirs'])}")
    if result["per_dir_counts"]:
        print("Files by input dir:")
        for dir_name, count in sorted(result["per_dir_counts"].items()):
            print(f"  {dir_name}: {count:,}")
    print(f"Competition-only count:  {result['competition_only_count']:,}")
    print(f"Total in business folder: {result['total_business_folder']:,}")
    if result["total_business_folder"]:
        pct = 100 * result["competition_only_count"] / result["total_business_folder"]
        print(f"Competition-only share:   {pct:.1f}%")
    print("----------------------------------------------------------\n")

    if args.list and result["competition_only_files"]:
        print("Competition-only filenames:")
        for name in sorted(result["competition_only_files"]):
            print(f"  {name}")


if __name__ == "__main__":
    main()