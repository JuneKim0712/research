#!/usr/bin/env python3
"""
One-command runner for prefilter working + cleaned outputs.

Runs:
  1) build_union_prefilter_output.py
  2) clean_prefilter_columns.py

Default mode tries to discover 2023/2024 mention-list inputs automatically.
You can also pass explicit input CSV paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from clean_prefilter_columns import clean_csv


def discover_inputs_for_year(year: str, root: Path) -> list[Path]:
    candidates = [
        root / f"ORG_detection/{year}/explicit_mentions_raw_nonraw.csv",
        root / f"ORG_detection_{year}/explicit_mentions_raw_nonraw.csv",
        root / f"{year}_explicit_mentions_raw_nonraw.csv",
        root / f"{year}/explicit_mentions_raw_nonraw.csv",
    ]

    found: list[Path] = [p for p in candidates if p.is_file()]
    if found:
        return found

    # Flexible fallback search.
    pattern = f"**/*{year}*explicit_mentions_raw_nonraw*.csv"
    globbed = sorted(p for p in root.glob(pattern) if p.is_file())
    return globbed[:1]


def run_build_step(python_exe: str, script_path: Path, input_csv: Path, output_csv: Path) -> None:
    cmd = [
        python_exe,
        str(script_path),
        "-i",
        str(input_csv),
        "-o",
        str(output_csv),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prefilter working+cleaned export in one command."
    )
    parser.add_argument(
        "--input",
        nargs="*",
        type=Path,
        help="Explicit input CSV path(s). If omitted, auto-discover by --years.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=["2023", "2024"],
        help="Years to auto-discover when --input is omitted (default: 2023 2024).",
    )
    parser.add_argument(
        "--working-name",
        default="union_filtered_prefilter_output.csv",
        help="Filename for the prefilter working output.",
    )
    parser.add_argument(
        "--cleaned-name",
        default="union_filtered_prefilter_output_cleaned.csv",
        help="Filename for the cleaned output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional fixed output directory for all outputs. Defaults to each input file's directory.",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    build_script = root / "build_union_prefilter_output.py"
    if not build_script.is_file():
        raise SystemExit(f"Missing script: {build_script}")

    inputs: list[Path] = []
    if args.input:
        inputs = [p.resolve() for p in args.input]
    else:
        seen: set[Path] = set()
        for year in args.years:
            for p in discover_inputs_for_year(str(year), root):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    inputs.append(rp)

    if not inputs:
        raise SystemExit(
            "No input files found. Pass --input explicitly or place year-tagged "
            "explicit_mentions_raw_nonraw CSV files in the workspace."
        )

    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"Input not found: {p}")

    print(f"Using Python: {sys.executable}")
    print(f"Found {len(inputs)} input file(s).")

    processed = 0
    for input_csv in inputs:
        target_dir = args.output_dir.resolve() if args.output_dir else input_csv.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        working_csv = target_dir / args.working_name
        cleaned_csv = target_dir / args.cleaned_name

        print("\n" + "-" * 80)
        print(f"Input:   {input_csv}")
        print(f"Working: {working_csv}")
        print(f"Cleaned: {cleaned_csv}")

        run_build_step(sys.executable, build_script, input_csv, working_csv)
        clean_csv(str(working_csv), str(cleaned_csv))

        processed += 1

    print("\nDone.")
    print(f"Processed files: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
