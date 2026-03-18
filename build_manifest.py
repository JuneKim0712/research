"""
build_manifest.py
=================
Scans a folder of extracted 10-K filing text files (e.g. {year}_10k_business/)
and produces a clean manifest in both CSV and JSON format, plus a parse-issues
audit file.

Expected filename format:
    YYYY-MM-DD__COMPANY NAME__ACCESSION_sectiontype.txt
    e.g. 2024-01-31__COMCAST CORP__0001166691-24-000011_business.txt

Usage:
    python build_manifest.py --input-dir 2024_10k_business
    python build_manifest.py --input-dir 2024_10k_business --output-dir ./output
    python build_manifest.py --input-dir 2024_10k_business --tiny-threshold 200

Outputs:
    manifest.csv
    manifest.json
    manifest_parse_issues.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches the leading date  YYYY-MM-DD
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

# Matches a canonical SEC accession number  ##########-##-######
ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")

# Accession numbers normalised to digits-only used for CIK extraction
CIK_FROM_ACCESSION_RE = re.compile(r"^(\d{10})")

# Characters that should be treated as segment separators in the stem when
# the double-underscore delimiter is not found.
FALLBACK_SEP_RE = re.compile(r"_{2,}")

# Collapses runs of whitespace / underscores in company name segment
CLEANUP_RE = re.compile(r"[_]{2,}|\s{2,}")

# Minimum character count to be considered non-empty/non-tiny
DEFAULT_TINY_THRESHOLD = 300  # chars

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _split_stem(stem: str) -> list[str]:
    """Split a file stem on double-underscores, falling back to single-segment."""
    if "__" in stem:
        return [s.strip() for s in stem.split("__")]
    # Try to split on single underscores after the date block
    parts = FALLBACK_SEP_RE.split(stem, maxsplit=2)
    return [s.strip() for s in parts]


def _parse_date(segment: str) -> Optional[str]:
    """Return YYYY-MM-DD if the segment starts with a valid date, else None."""
    m = DATE_RE.match(segment.strip())
    if not m:
        return None
    raw = m.group(1)
    # Basic sanity: month 01-12, day 01-31
    parts = raw.split("-")
    if len(parts) != 3:
        return None
    _, mm, dd = parts
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31):
        return None
    return raw


def _parse_accession_and_section(segment: str) -> tuple[Optional[str], Optional[str]]:
    """
    From a segment like '0001166691-24-000011_business', extract:
        accession_number -> '0001166691-24-000011'
        section_type     -> 'business'
    The section type is everything after the last underscore that follows the
    accession number.
    """
    m = ACCESSION_RE.search(segment)
    if not m:
        return None, None

    accession = m.group(1)
    # Whatever comes after the accession match, strip leading/trailing underscores
    after = segment[m.end():].strip("_ \t")
    section_type = after if after else None
    # Normalise: lowercase, strip trailing noise
    if section_type:
        section_type = re.sub(r"[^a-z0-9_-]", "", section_type.lower()).strip("_-")
        section_type = section_type if section_type else None

    return accession, section_type


def _cik_from_accession(accession: str) -> tuple[Optional[str], Optional[str]]:
    """
    Return (cik_raw, cik_int_str) where:
        cik_raw     preserves leading zeros  e.g. '0001166691'
        cik_int_str is the integer representation  e.g. '1166691'
    """
    digits_only = accession.replace("-", "")
    m = CIK_FROM_ACCESSION_RE.match(digits_only)
    if not m:
        return None, None
    raw = m.group(1)          # 10 chars, may have leading zeros
    return raw, str(int(raw)) # strip leading zeros for integer form


def _clean_company_name(raw: str) -> str:
    """
    Light cleaning of the company name segment:
      - Replace underscores with spaces
      - Collapse multiple spaces/underscores
      - Strip outer whitespace
      - Convert to title-case only when the name is ALL-CAPS (preserve mixed)
    """
    name = raw.replace("_", " ")
    name = CLEANUP_RE.sub(" ", name)
    name = name.strip()
    # If entirely uppercase (common in SEC data), title-case for readability
    # but preserve if already mixed-case (some names use intentional caps)
    if name == name.upper():
        name = name.title()
    return name


def parse_filename(stem: str) -> dict:
    """
    Parse as much metadata as possible from the file stem.

    Returns a dict with keys:
        filing_date, filing_year, source_company_name,
        accession_number, cik_raw, source_cik,
        section_type, parse_success, parse_notes
    """
    notes: list[str] = []
    result: dict = {
        "filing_date": None,
        "filing_year": None,
        "source_company_name": None,
        "accession_number": None,
        "cik_raw": None,
        "source_cik": None,
        "section_type": None,
        "parse_success": True,
        "parse_notes": "",
    }

    parts = _split_stem(stem)

    # ── 1. Filing date (first segment) ──────────────────────────────────────
    if parts:
        date = _parse_date(parts[0])
        if date:
            result["filing_date"] = date
            result["filing_year"] = date[:4]
        else:
            notes.append(f"date not found in first segment '{parts[0]}'")
            result["parse_success"] = False

    # ── 2. Company name (middle segment when 3+ parts available) ────────────
    if len(parts) >= 3:
        result["source_company_name"] = _clean_company_name(parts[1])
    elif len(parts) == 2:
        # Could be date + accession_section; try to grab company from nowhere
        notes.append("only 2 segments found; company name may be missing")
        # Check if the second segment looks like an accession
        if not ACCESSION_RE.search(parts[1]):
            result["source_company_name"] = _clean_company_name(parts[1])
        else:
            result["parse_success"] = False
    else:
        notes.append("could not identify company name segment")
        result["parse_success"] = False

    # ── 3. Accession number + section type (last segment) ───────────────────
    last_segment = parts[-1] if parts else ""
    accession, section_type = _parse_accession_and_section(last_segment)

    if accession:
        result["accession_number"] = accession
        cik_raw, cik_int = _cik_from_accession(accession)
        result["cik_raw"] = cik_raw
        result["source_cik"] = cik_int
    else:
        notes.append(f"accession number not found in last segment '{last_segment}'")
        result["parse_success"] = False

    if section_type:
        result["section_type"] = section_type
    else:
        # Accession was found but section suffix missing — try whole stem
        leftover = re.sub(ACCESSION_RE, "", last_segment).strip("_- \t")
        if leftover:
            result["section_type"] = leftover
            notes.append(f"section_type inferred from suffix: '{leftover}'")
        else:
            notes.append("section_type not found")
            result["parse_success"] = False

    result["parse_notes"] = "; ".join(notes) if notes else ""
    return result


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def read_text_robust(path: Path) -> str:
    """Try common encodings before falling back to ignore-mode."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def content_fallback(text: str, field: str) -> Optional[str]:
    """
    Lightweight fallback – scans the first 4 KB of file content to try to
    recover a field that could not be parsed from the filename.

    Supported fields:
        'section_type'  – looks for a line like "Section: business"
                          or the first Item 1 heading

    No NLP. Only simple regex on the file header.
    """
    if field == "section_type":
        head = text[:4096]
        m = re.search(r"(?i)(?:section[_\s]*type|section)\s*[:\-]\s*(\w+)", head)
        if m:
            return m.group(1).lower()
        # Check for Item 1 / competition heading presence
        if re.search(r"(?mi)^\s*item\s*1\s*[.\-:]?\s*business\b", head):
            return "business"
        if re.search(r"(?mi)^\s*item\s*1\s*[.\-:]?\s*(our\s+)?competition\b", head):
            return "competition"
    return None


# ---------------------------------------------------------------------------
# Core manifest builder
# ---------------------------------------------------------------------------

def build_manifest_row(path: Path, tiny_threshold: int) -> dict:
    """Build one manifest row for the given .txt file."""
    stem = path.stem
    filename = path.name
    file_path = str(path.resolve())

    # ── Parse metadata from filename ────────────────────────────────────────
    parsed = parse_filename(stem)

    # ── Read content ─────────────────────────────────────────────────────────
    try:
        text = read_text_robust(path)
        char_count = len(text)
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        is_empty_or_tiny = char_count < tiny_threshold
    except Exception as exc:
        text = ""
        char_count = 0
        line_count = 0
        is_empty_or_tiny = True
        parsed["parse_success"] = False
        note = f"file read error: {exc}"
        parsed["parse_notes"] = (parsed["parse_notes"] + "; " + note).lstrip("; ")

    # ── Content fallback for missing fields ──────────────────────────────────
    if not parsed["section_type"] and text:
        fallback_st = content_fallback(text, "section_type")
        if fallback_st:
            parsed["section_type"] = fallback_st
            note = f"section_type from content fallback: '{fallback_st}'"
            parsed["parse_notes"] = (parsed["parse_notes"] + "; " + note).lstrip("; ")

    # ── Assemble row (ordered for readability) ───────────────────────────────
    return {
        # Essential fields
        "source_company_name": parsed["source_company_name"],
        "source_cik":          parsed["source_cik"],
        "filing_year":         parsed["filing_year"],
        "filing_date":         parsed["filing_date"],
        "accession_number":    parsed["accession_number"],
        "original_filename":   filename,
        "section_type":        parsed["section_type"],
        # Helper fields
        "file_path":           file_path,
        "file_stem":           stem,
        "text_char_count":     char_count,
        "text_line_count":     line_count,
        "is_empty_or_tiny":    is_empty_or_tiny,
        "parse_success":       parsed["parse_success"],
        "parse_notes":         parsed["parse_notes"],
        # Extra parsed detail (bonus, non-essential)
        "cik_raw":             parsed["cik_raw"],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

ESSENTIAL_FIELDS = [
    "source_company_name",
    "source_cik",
    "filing_year",
    "filing_date",
    "accession_number",
    "original_filename",
    "section_type",
]

HELPER_FIELDS = [
    "file_path",
    "file_stem",
    "text_char_count",
    "text_line_count",
    "is_empty_or_tiny",
    "parse_success",
    "parse_notes",
    "cik_raw",
]

CSV_FIELDNAMES = ESSENTIAL_FIELDS + HELPER_FIELDS


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_parse_issues(rows: list[dict], out_path: Path) -> None:
    issues = [
        r for r in rows
        if not r["parse_success"] or r["parse_notes"]
    ]
    lines: list[str] = [
        "manifest_parse_issues.txt",
        "=" * 60,
        f"Files with one or more parse issues: {len(issues)}",
        "",
    ]
    for r in issues:
        lines.append(f"FILE: {r['original_filename']}")
        for field in ESSENTIAL_FIELDS:
            val = r.get(field)
            if val is None or val == "":
                lines.append(f"  MISSING: {field}")
        if r["parse_notes"]:
            lines.append(f"  NOTES:   {r['parse_notes']}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict], input_dir: Path) -> None:
    total = len(rows)
    ok = sum(1 for r in rows if r["parse_success"])
    failed = total - ok
    tiny = sum(1 for r in rows if r["is_empty_or_tiny"])
    section_types = sorted({r["section_type"] for r in rows if r["section_type"]})

    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  MANIFEST SUMMARY  —  {input_dir}")
    print(sep)
    print(f"  Total .txt files found     : {total:>6,}")
    print(f"  Successfully parsed        : {ok:>6,}")
    print(f"  Failed / partial parses    : {failed:>6,}")
    print(f"  Empty or tiny files        : {tiny:>6,}  (< {DEFAULT_TINY_THRESHOLD} chars)")
    print(f"  Distinct section types     : {len(section_types):>6,}")
    if section_types:
        for st in section_types:
            count = sum(1 for r in rows if r["section_type"] == st)
            print(f"    • {st:<30} {count:>6,}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a manifest CSV/JSON from a folder of extracted 10-K text files."
    )
    p.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Path to the folder containing extracted *_business.txt files "
             "(e.g. 2024_10k_business/).",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory where manifest files are written (default: same as input-dir).",
    )
    p.add_argument(
        "--tiny-threshold",
        type=int,
        default=DEFAULT_TINY_THRESHOLD,
        help=f"Files with fewer than this many characters are flagged as "
             f"empty/tiny (default: {DEFAULT_TINY_THRESHOLD}).",
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=False,
        help="Also scan subdirectories inside input-dir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        sys.exit(f"[ERROR] Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect .txt files ──────────────────────────────────────────────────
    glob_fn = input_dir.rglob if args.recursive else input_dir.glob
    txt_files = sorted(glob_fn("*.txt"))

    if not txt_files:
        sys.exit(f"[ERROR] No .txt files found in: {input_dir}")

    print(f"[build_manifest] Scanning {len(txt_files):,} .txt files in: {input_dir}")

    # ── Build rows ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    for path in txt_files:
        row = build_manifest_row(path, tiny_threshold=args.tiny_threshold)
        rows.append(row)

    # ── Write outputs ───────────────────────────────────────────────────────
    csv_path   = output_dir / "manifest.csv"
    json_path  = output_dir / "manifest.json"
    audit_path = output_dir / "manifest_parse_issues.txt"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_parse_issues(rows, audit_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(rows, input_dir)
    print(f"\n  Outputs written to: {output_dir}")
    print(f"    • manifest.csv")
    print(f"    • manifest.json")
    print(f"    • manifest_parse_issues.txt")
    print()


if __name__ == "__main__":
    main()