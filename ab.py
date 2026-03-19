#!/usr/bin/env python3
"""
Append competition sections to business files when both exist in the cleaned file.

For each file in the cleaned folder that has BOTH a business section AND a
competition section, extracts the competition section (200–50k chars) and
appends it to the bottom of the matching company's business file.

Inspired by a.py extraction logic. Self-contained (no a.py import) for portability.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm

YEAR_CLEANED_DIR_RE = re.compile(r"^(?P<year>\d{4})_10k_cleaned$", flags=re.I)

# ── Patterns (from a.py) ────────────────────────────────────────────────────
BUSINESS_START_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("item1_dot_business", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]\s*business\b")),
    ("item1_our_business", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]?\s*(our|the|company's)?\s*business\b")),
    ("item_one_business", re.compile(r"(?mi)^\s*item\s+(one|1st)\s*[.\-\u2013\u2014:]?\s*business\b")),
    ("item1_desc_business", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]?\s*description\s+of\s+(?:our\s+)?business\b")),
    ("item1_general", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]\s*general\s*$")),
    ("item1_overview", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]\s*overview\s*$")),
    ("our_business", re.compile(r"(?mi)^\s*our\s+business\s*$")),
    ("business_overview", re.compile(r"(?mi)^\s*business\s+overview\s*$")),
    ("company_overview", re.compile(r"(?mi)^\s*company\s+overview\s*$")),
    ("overview_of_business", re.compile(r"(?mi)^\s*overview\s+of\s+(?:our\s+)?business\s*$")),
    ("bare_business", re.compile(r"(?mi)^\s*business\s*$")),
]

BUSINESS_END_RES = [
    re.compile(r"(?mi)^\s*item\s*1\s*a\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*1\s*b\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*1\s*c\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*2\s*[.\-\u2013\u2014:]"),
    re.compile(r"(?mi)^\s*item\s*3\s*[.\-\u2013\u2014:]"),
    re.compile(r"(?mi)^\s*part\s*(?:ii|2)\s*[.\-\u2013\u2014]?"),
]

COMPETITION_START_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("item1_competition", re.compile(r"(?mi)^\s*item\s*1\s*[.\-\u2013\u2014:]?\s*(our\s+)?competition\b")),
    ("our_competition", re.compile(r"(?mi)^\s*our\s+competition\s*$")),
    ("competition", re.compile(r"(?mi)^\s*competition\s*$")),
]

COMPETITION_END_RES = [
    re.compile(r"(?mi)^\s*item\s*1\s*a\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*1\s*b\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*1\s*c\s*[.\-\u2013\u2014:]?"),
    re.compile(r"(?mi)^\s*item\s*2\s*[.\-\u2013\u2014:]"),
    re.compile(r"(?mi)^\s*item\s*3\s*[.\-\u2013\u2014:]"),
    re.compile(r"(?mi)^\s*part\s*(?:ii|2)\s*[.\-\u2013\u2014]?"),
]


def read_text_robust(path: Path) -> str:
    """Read file with multiple encoding fallbacks."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_section_by_patterns(
    text: str,
    start_patterns: list[tuple[str, re.Pattern]],
    end_patterns: list[re.Pattern],
    min_len: int,
    max_len: int = 500_000,
) -> Optional[str]:
    """Extract section matching start/end patterns (from a.py)."""
    if not any(start_re.search(text) for _, start_re in start_patterns):
        return None

    best: Optional[str] = None
    for _label, start_re in start_patterns:
        candidates = []
        for m in start_re.finditer(text):
            start = m.start()
            surrounding = text[max(0, start - 5) : m.end() + 20]
            if re.search(r"\.{3,}|\t\d{1,3}\s*$", surrounding, flags=re.I):
                continue

            ends = []
            for end_re in end_patterns:
                end_m = end_re.search(text, pos=m.end() + 1)
                if end_m and end_m.start() > start:
                    ends.append(end_m.start())

            end = min(ends) if ends else len(text)
            section = text[start:end].strip()
            if min_len <= len(section) <= max_len:
                candidates.append(section)

        if candidates:
            local_best = max(candidates, key=len)
            if best is None or len(local_best) > len(best):
                best = local_best
            break

    return best

# Competition section bounds: min 200, max 50k chars
COMPETITION_MIN_CHARS = 200
COMPETITION_MAX_CHARS = 50_000

# Separator between business content and appended competition section
COMPETITION_APPEND_SEPARATOR = "\n\n--- ITEM 1 – COMPETITION (appended) ---\n\n"


def extract_business_section(cleaned_text: str) -> Optional[str]:
    """Extract business section (min 500 chars, same as a.py)."""
    return extract_section_by_patterns(
        cleaned_text,
        BUSINESS_START_PATTERNS,
        BUSINESS_END_RES,
        min_len=500,
    )


def extract_competition_section(cleaned_text: str) -> Optional[str]:
    """Extract competition section (min 200, max 50k chars)."""
    return extract_section_by_patterns(
        cleaned_text,
        COMPETITION_START_PATTERNS,
        COMPETITION_END_RES,
        min_len=COMPETITION_MIN_CHARS,
        max_len=COMPETITION_MAX_CHARS,
    )


def process_cleaned_file(
    cleaned_path: Path,
    business_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    If cleaned file has both business and competition sections, append
    competition to the matching business file. Returns status dict.
    """
    cleaned_text = read_text_robust(cleaned_path)
    business_section = extract_business_section(cleaned_text)
    competition_section = extract_competition_section(cleaned_text)

    stem = cleaned_path.stem
    business_path = business_dir / f"{stem}_business.txt"

    if business_section is None or competition_section is None:
        return {"status": "skipped", "reason": "missing_section", "path": str(cleaned_path)}
    if not business_path.exists():
        return {"status": "skipped", "reason": "no_business_file", "path": str(cleaned_path)}

    # Check if competition already appended (avoid duplicate appends)
    current_content = read_text_robust(business_path)
    if COMPETITION_APPEND_SEPARATOR in current_content:
        return {"status": "skipped", "reason": "already_appended", "path": str(cleaned_path)}

    if not dry_run:
        new_content = current_content.rstrip() + COMPETITION_APPEND_SEPARATOR + competition_section.strip()
        business_path.write_text(new_content, encoding="utf-8")

    return {
        "status": "appended",
        "path": str(cleaned_path),
        "business_file": str(business_path),
        "competition_chars": len(competition_section),
    }


def run(
    cleaned_dir: Path,
    business_dir: Path,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Process all cleaned files, append competition to business when applicable."""
    cleaned_dir = cleaned_dir.resolve()
    business_dir = business_dir.resolve()

    if not cleaned_dir.exists():
        raise FileNotFoundError(f"Cleaned directory not found: {cleaned_dir}")
    business_dir.mkdir(parents=True, exist_ok=True)

    cleaned_files = sorted(cleaned_dir.glob("*.txt"))
    if limit:
        cleaned_files = cleaned_files[:limit]

    appended = 0
    skipped_missing = 0
    skipped_no_file = 0
    skipped_already = 0

    progress_desc = f"{cleaned_dir.name}"
    for cleaned_path in tqdm(cleaned_files, desc=progress_desc, unit="file"):
        result = process_cleaned_file(cleaned_path, business_dir, dry_run=dry_run)
        if result["status"] == "appended":
            appended += 1
        elif result["reason"] == "missing_section":
            skipped_missing += 1
        elif result["reason"] == "no_business_file":
            skipped_no_file += 1
        elif result["reason"] == "already_appended":
            skipped_already += 1

    return {
        "total_cleaned": len(cleaned_files),
        "appended": appended,
        "skipped_missing_section": skipped_missing,
        "skipped_no_business_file": skipped_no_file,
        "skipped_already_appended": skipped_already,
    }


def derive_business_dir_from_cleaned(cleaned_dir: Path) -> Path:
    """Map {year}_10K_cleaned to {year}_10K_business in the same parent directory."""
    match = YEAR_CLEANED_DIR_RE.match(cleaned_dir.name)
    if not match:
        raise ValueError(
            f"Could not infer year from cleaned dir '{cleaned_dir.name}'. "
            "Expected format like 2023_10K_cleaned."
        )
    return cleaned_dir.parent / f"{match.group('year')}_10K_business"


def discover_cleaned_dirs(base_dir: Path, year: Optional[int] = None) -> list[Path]:
    """Find cleaned directories named {year}_10K_cleaned (case-insensitive k)."""
    base_dir = base_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if year is not None:
        candidates = [base_dir / f"{year}_10K_cleaned", base_dir / f"{year}_10k_cleaned"]
        existing = []
        for path in candidates:
            if path.exists() and path.is_dir() and path not in existing:
                existing.append(path)
        return existing

    return sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir() and YEAR_CLEANED_DIR_RE.match(path.name)
    )


def run_across_cleaned_dirs(
    cleaned_dirs: list[Path],
    business_dir_override: Optional[Path],
    limit: Optional[int],
    dry_run: bool,
) -> dict:
    """Run processing for one or more cleaned dirs and aggregate summary stats."""
    total = {
        "dirs_processed": 0,
        "total_cleaned": 0,
        "appended": 0,
        "skipped_missing_section": 0,
        "skipped_no_business_file": 0,
        "skipped_already_appended": 0,
    }

    for cleaned_dir in cleaned_dirs:
        business_dir = business_dir_override or derive_business_dir_from_cleaned(cleaned_dir)
        stats = run(cleaned_dir, business_dir, limit=limit, dry_run=dry_run)

        total["dirs_processed"] += 1
        total["total_cleaned"] += stats["total_cleaned"]
        total["appended"] += stats["appended"]
        total["skipped_missing_section"] += stats["skipped_missing_section"]
        total["skipped_no_business_file"] += stats["skipped_no_business_file"]
        total["skipped_already_appended"] += stats["skipped_already_appended"]

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append competition sections to business files when both exist in cleaned file."
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=None,
        help="Single cleaned folder to process (e.g., 2023_10K_cleaned)",
    )
    parser.add_argument(
        "--business-dir",
        type=Path,
        default=None,
        help="Optional business folder override for all input files",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base folder to discover year cleaned dirs (default: current directory)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional year filter for discovered dirs (e.g., 2023)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be appended without modifying files",
    )
    args = parser.parse_args()

    if args.cleaned_dir is not None:
        cleaned_dirs = [args.cleaned_dir.resolve()]
    else:
        cleaned_dirs = discover_cleaned_dirs(args.base_dir, args.year)

    if not cleaned_dirs:
        if args.year is None:
            raise FileNotFoundError("No cleaned directories found matching *_10K_cleaned")
        raise FileNotFoundError(f"No cleaned directory found for year {args.year}")

    stats = run_across_cleaned_dirs(
        cleaned_dirs,
        business_dir_override=args.business_dir.resolve() if args.business_dir else None,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    mode = " (dry run)" if args.dry_run else ""
    print(f"\n--- Append Competition to Business Summary{mode} ---")
    print(f"Year cleaned dirs processed   : {stats['dirs_processed']:,}")
    print(f"Total cleaned files processed : {stats['total_cleaned']:,}")
    print(f"Competition appended         : {stats['appended']:,}")
    print(f"Skipped (missing section)     : {stats['skipped_missing_section']:,}")
    print(f"Skipped (no business file)   : {stats['skipped_no_business_file']:,}")
    print(f"Skipped (already appended)   : {stats['skipped_already_appended']:,}")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()