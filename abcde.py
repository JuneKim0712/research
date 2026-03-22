"""
Build explicit_candidate_windows.csv for downstream explicit mention extraction.

Reads ABCD bucket exports (candidate_windows_strict_explicit.*,
candidate_windows_contextual_explicit.*) or a combined candidate_windows CSV/JSONL
(from abcd.py or build_candidate_windows.py). Retains only strict/contextual
explicit rows, maps to a stable column schema, drops exact duplicate evidence
rows (same identity + text fields), and assigns unique sequential window_ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

EXPLICIT_BUCKETS = frozenset({"strict_explicit", "contextual_explicit"})
EXPLICIT_CUE_GROUPS = frozenset({"strict_explicit", "contextual_explicit"})

INPUT_COLUMN_ALIASES = {
    "source_company": ("source_company", "source_company_name"),
    "section": ("section", "section_type"),
    "heading": ("heading", "heading_text"),
    "cue_phrase": ("cue_phrase", "cue_text"),
}

OUTPUT_FIELDS = [
    "window_id",
    "source_company",
    "source_cik",
    "submitter_cik",
    "filing_year",
    "accession_number",
    "filing_id",
    "section",
    "heading",
    "cue_phrase",
    "cue_group",
    "trigger_sentence",
    "window_text",
    "window_priority",
    "export_bucket",
]


def infer_export_bucket(cue_group: str) -> str:
    if cue_group == "strict_explicit":
        return "strict_explicit"
    if cue_group == "contextual_explicit":
        return "contextual_explicit"
    if cue_group == "implicit_or_broad":
        return "broad_or_implicit"
    if cue_group == "heading_fallback_broad":
        return "broad_or_implicit"
    if cue_group == "demotion_or_negative":
        return "broad_or_implicit"
    return ""


def is_explicit_window(row: dict[str, Any]) -> bool:
    cg = str(row.get("cue_group") or "").strip()
    if cg in EXPLICIT_CUE_GROUPS:
        return True
    eb = str(row.get("export_bucket") or "").strip()
    if eb in EXPLICIT_BUCKETS:
        return True
    return False


def _coerce_priority(val: Any) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, bool):
        return str(int(val))
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return str(val).strip()


def _pick_value(raw: dict[str, Any], key: str) -> str:
    for alias in INPUT_COLUMN_ALIASES.get(key, (key,)):
        v = raw.get(alias)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def normalize_row(raw: dict[str, Any]) -> dict[str, str]:
    company = _pick_value(raw, "source_company")
    acc = raw.get("accession_number") or ""
    fid = raw.get("filing_id") or ""
    cue_group = str(raw.get("cue_group") or "").strip()
    export_bucket = str(raw.get("export_bucket") or "").strip()
    if not export_bucket and cue_group:
        export_bucket = infer_export_bucket(cue_group)

    return {
        "source_company": str(company).strip(),
        "source_cik": str(raw.get("source_cik") or "").strip(),
        "submitter_cik": str(raw.get("submitter_cik") or "").strip(),
        "filing_year": str(raw.get("filing_year") or "").strip(),
        "accession_number": str(acc).strip(),
        "filing_id": str(fid).strip(),
        "section": _pick_value(raw, "section"),
        "heading": _pick_value(raw, "heading"),
        "cue_phrase": _pick_value(raw, "cue_phrase"),
        "cue_group": cue_group,
        "trigger_sentence": str(raw.get("trigger_sentence") or ""),
        "window_text": str(raw.get("window_text") or ""),
        "window_priority": _coerce_priority(raw.get("window_priority")),
        "export_bucket": export_bucket,
    }


def exact_dedupe_key(row: dict[str, str]) -> tuple:
    """All substantive fields except window_id — drop only true duplicate evidence."""
    return tuple(row[k] for k in OUTPUT_FIELDS if k != "window_id")


def sort_key(row: dict[str, str]) -> tuple:
    try:
        p = int(row["window_priority"]) if row["window_priority"] else 0
    except ValueError:
        p = 0
    return (
        row["filing_year"],
        row["source_company"].lower(),
        row["source_cik"],
        row["accession_number"],
        row["filing_id"],
        -p,
        row["trigger_sentence"][:120],
        row["window_text"][:120],
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    
    iterator = tqdm(lines, desc=f"Loading {path.name}") if tqdm else lines
    for line in iterator:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_any(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl(path)
    return load_csv(path)


def default_input_paths(cwd: Path) -> list[Path]:
    strict = cwd / "candidate_windows_strict_explicit.csv"
    ctx = cwd / "candidate_windows_contextual_explicit.csv"
    if strict.is_file() or ctx.is_file():
        out: list[Path] = []
        if strict.is_file():
            out.append(strict)
        if ctx.is_file():
            out.append(ctx)
        return out
    jl = cwd / "candidate_windows.jsonl"
    if jl.is_file():
        return [jl]
    csv_path = cwd / "candidate_windows.csv"
    if csv_path.is_file():
        return [csv_path]
    return []


def discover_inputs(args_inputs: list[Path], cwd: Path) -> list[Path]:
    if args_inputs:
        return [p.resolve() for p in args_inputs]
    found = default_input_paths(cwd)
    if not found:
        raise SystemExit(
            "No inputs given and no default candidate window files found. "
            "Pass CSV/JSONL paths or place candidate_windows*.csv/jsonl in the working directory."
        )
    return found


def build_explicit_rows(sources: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    iterator = tqdm(sources, desc="Normalizing rows") if tqdm else sources
    for raw in iterator:
        if not is_explicit_window(raw):
            continue
        normalized.append(normalize_row(raw))

    normalized.sort(key=sort_key)

    seen: set[tuple] = set()
    deduped: list[dict[str, str]] = []
    dedup_iterator = tqdm(normalized, desc="Deduplicating rows") if tqdm else normalized
    for row in dedup_iterator:
        k = exact_dedupe_key(row)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)

    # Add window IDs
    window_id_iterator = tqdm(enumerate(deduped, start=1), desc="Assigning window IDs", total=len(deduped)) if tqdm else enumerate(deduped, start=1)
    for i, row in window_id_iterator:
        row["window_id"] = f"EXPW{i:07d}"

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter explicit candidate windows into explicit_candidate_windows.csv",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input CSV/JSONL files (default: bucket CSVs or candidate_windows.*)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("explicit_candidate_windows.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Directory for default input discovery (default: current directory)",
    )
    args = parser.parse_args()

    cwd = (args.cwd or Path(".")).resolve()
    paths = discover_inputs(list(args.inputs), cwd)

    all_raw: list[dict[str, Any]] = []
    iterator = tqdm(paths, desc="Loading input files") if tqdm else paths
    for p in iterator:
        if not p.is_file():
            raise SystemExit(f"Missing input file: {p}")
        all_raw.extend(load_any(p))

    rows = build_explicit_rows(all_raw)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Inputs ({len(paths)}): " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()