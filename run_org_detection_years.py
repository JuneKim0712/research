#!/usr/bin/env python3
"""
Run ORG-detection pipeline for one or more filing years.

Pipeline per year:
  1) build_manifest.py on {year}_10K_business
  2) abcd.py -> {year}_abcd (as requested)
  3) abcde.py -> ORG_detection/{year}_explicit_candidate_windows.csv
  4) abcdef.py -> ORG_detection/{year}_explicit_mentions_raw*.csv
  5) build_union_prefilter_output.py -> ORG_detection/{year}_union_filtered_prefilter_output.csv
  6) clean_prefilter_columns.py -> ORG_detection/{year}_union_filtered_prefilter_output_cleaned.csv
  7) Copy/flatten CSV artifacts into ORG_detection with {year}_ prefixes

Progress bars:
  - Year-level progress bar
  - Step-level progress bar for every major step
  - Substep progress bar for each step (prep/run/verify/copy)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class YearPaths:
    year: str
    business_dir: Path
    manifest_dir: Path
    manifest_csv: Path
    abcd_dir: Path
    explicit_csv: Path
    raw_csv: Path
    nonraw_csv: Path
    org_diff_csv: Path
    prefilter_csv: Path
    cleaned_csv: Path


def _bar(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit="sub",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )


def bar_update(bar, n: int = 1) -> None:
    if bar is not None:
        bar.update(n)


def bar_close(bar) -> None:
    if bar is not None:
        bar.close()


def run_cmd(cmd: list[str], cwd: Path, step_desc: str) -> None:
    print(f"\n[{step_desc}] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{step_desc} failed with exit code {proc.returncode}")


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected {label}: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing expected {label}: {path}")


def copy_with_prefix(files: Iterable[Path], out_dir: Path, prefix: str) -> list[Path]:
    copied: list[Path] = []
    for src in files:
        if not src.is_file():
            continue
        dest = out_dir / f"{prefix}_{src.name}"
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def build_year_paths(root: Path, out_dir: Path, year: str) -> YearPaths:
    business_candidates = [
        root / f"{year}_10K_business",
        root / f"{year}_10k_business",
        root / year / f"{year}_10K_business",
        root / year / f"{year}_10k_business",
    ]
    business_dir = next((p for p in business_candidates if p.is_dir()), business_candidates[0])

    return YearPaths(
        year=year,
        business_dir=business_dir,
        manifest_dir=out_dir / f"{year}_manifest",
        manifest_csv=out_dir / f"{year}_manifest" / "manifest.csv",
        abcd_dir=root / f"{year}_abcd",
        explicit_csv=out_dir / f"{year}_explicit_candidate_windows.csv",
        raw_csv=out_dir / f"{year}_explicit_mentions_raw.csv",
        nonraw_csv=out_dir / f"{year}_explicit_mentions_raw_nonraw.csv",
        org_diff_csv=out_dir / f"{year}_explicit_mentions_raw_org_diff.csv",
        prefilter_csv=out_dir / f"{year}_union_filtered_prefilter_output.csv",
        cleaned_csv=out_dir / f"{year}_union_filtered_prefilter_output_cleaned.csv",
    )


def run_year(root: Path, py: str, paths: YearPaths, out_dir: Path) -> None:
    ensure_dir(paths.business_dir, f"{paths.year} business input folder")

    # Step 1: Manifest
    step = _bar(4, f"{paths.year} Step 1/7 manifest")
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "build_manifest.py",
            "--input-dir",
            str(paths.business_dir),
            "--output-dir",
            str(out_dir),
        ],
        cwd=root,
        step_desc=f"{paths.year} manifest",
    )
    bar_update(step)  # run
    ensure_file(paths.manifest_csv, "manifest.csv")
    bar_update(step)  # verify
    # Flatten manifest CSV to ORG_detection/{year}_manifest.csv
    manifest_flat = out_dir / f"{paths.year}_manifest.csv"
    shutil.copy2(paths.manifest_csv, manifest_flat)
    bar_update(step)  # copy
    bar_close(step)

    # Step 2: ABCD in {year}_abcd
    step = _bar(4, f"{paths.year} Step 2/7 abcd")
    paths.abcd_dir.mkdir(parents=True, exist_ok=True)
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "abcd.py",
            str(paths.manifest_csv),
            "--base-dir",
            str(root),
            "--output-dir",
            str(paths.abcd_dir),
        ],
        cwd=root,
        step_desc=f"{paths.year} abcd",
    )
    bar_update(step)  # run

    abcd_expected_required = [
        paths.abcd_dir / "candidate_windows_strict_explicit.csv",
        paths.abcd_dir / "candidate_windows_contextual_explicit.csv",
        paths.abcd_dir / "candidate_windows_broad_or_implicit.csv",
    ]
    for p in abcd_expected_required:
        ensure_file(p, f"ABCD output {p.name}")

    abcd_csvs = sorted(p for p in paths.abcd_dir.glob("*.csv") if p.is_file())
    if not abcd_csvs:
        raise FileNotFoundError(f"No ABCD CSV outputs found in: {paths.abcd_dir}")
    bar_update(step)  # verify
    copy_with_prefix(abcd_csvs, out_dir, paths.year)
    bar_update(step)  # copy
    bar_close(step)

    # Step 3: Explicit windows
    step = _bar(4, f"{paths.year} Step 3/7 abcde")
    strict_csv = paths.abcd_dir / "candidate_windows_strict_explicit.csv"
    contextual_csv = paths.abcd_dir / "candidate_windows_contextual_explicit.csv"
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "abcde.py",
            str(strict_csv),
            str(contextual_csv),
            "-o",
            str(paths.explicit_csv),
        ],
        cwd=root,
        step_desc=f"{paths.year} explicit windows",
    )
    bar_update(step)  # run
    ensure_file(paths.explicit_csv, "explicit candidate windows CSV")
    bar_update(step)  # verify
    bar_update(step)  # copy/already in target dir
    bar_close(step)

    # Step 4: NER mentions
    step = _bar(4, f"{paths.year} Step 4/7 abcdef")
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "abcdef.py",
            "-i",
            str(paths.explicit_csv),
            "-o",
            str(paths.raw_csv),
            "--preview",
            "0",
        ],
        cwd=root,
        step_desc=f"{paths.year} NER mentions",
    )
    bar_update(step)  # run
    ensure_file(paths.raw_csv, "raw mention CSV")
    ensure_file(paths.nonraw_csv, "nonraw mention CSV")
    ensure_file(paths.org_diff_csv, "org diff CSV")
    bar_update(step)  # verify
    bar_update(step)  # copy/already in target dir
    bar_close(step)

    # Step 5: Prefilter working output
    step = _bar(4, f"{paths.year} Step 5/7 prefilter build")
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "build_union_prefilter_output.py",
            "-i",
            str(paths.nonraw_csv),
            "-o",
            str(paths.prefilter_csv),
        ],
        cwd=root,
        step_desc=f"{paths.year} prefilter build",
    )
    bar_update(step)  # run
    ensure_file(paths.prefilter_csv, "prefilter working CSV")
    bar_update(step)  # verify
    bar_update(step)  # copy/already in target dir
    bar_close(step)

    # Step 6: Clean prefilter output
    step = _bar(4, f"{paths.year} Step 6/7 prefilter clean")
    bar_update(step)  # prep
    run_cmd(
        [
            py,
            "clean_prefilter_columns.py",
            str(paths.prefilter_csv),
            "-o",
            str(paths.cleaned_csv),
        ],
        cwd=root,
        step_desc=f"{paths.year} prefilter clean",
    )
    bar_update(step)  # run
    ensure_file(paths.cleaned_csv, "prefilter cleaned CSV")
    bar_update(step)  # verify
    bar_update(step)  # copy/already in target dir
    bar_close(step)

    # Step 7: Flatten any remaining CSV artifacts for this year into ORG_detection
    step = _bar(4, f"{paths.year} Step 7/7 flatten csv")
    bar_update(step)  # prep
    extra_sources = [
        paths.manifest_dir / "manifest.csv",
        paths.manifest_dir / "manifest.json",  # ignored by copy helper
        paths.abcd_dir / "candidate_windows.jsonl",  # ignored by copy helper
    ]
    year_csvs = [p for p in paths.abcd_dir.glob("*.csv") if p.is_file()]
    copied = copy_with_prefix(year_csvs, out_dir, paths.year)
    bar_update(step)  # run(copy)
    # Verify key outputs exist in ORG_detection
    must_exist = [
        out_dir / f"{paths.year}_manifest.csv",
        out_dir / f"{paths.year}_candidate_windows_strict_explicit.csv",
        out_dir / f"{paths.year}_candidate_windows_contextual_explicit.csv",
        out_dir / f"{paths.year}_candidate_windows_broad_or_implicit.csv",
        paths.explicit_csv,
        paths.raw_csv,
        paths.nonraw_csv,
        paths.org_diff_csv,
        paths.prefilter_csv,
        paths.cleaned_csv,
    ]
    for p in must_exist:
        ensure_file(p, f"flattened/final CSV {p.name}")
    bar_update(step)  # verify
    bar_update(step)  # done
    bar_close(step)

    print(f"\n[{paths.year}] Completed. Copied {len(copied)} ABCD CSV files into {out_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2023/2024 ORG detection pipeline with progress bars and centralized CSV outputs."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2023", "2024"],
        help="Years to process (default: 2023 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ORG_detection"),
        help="Directory to collect all CSV outputs (default: ORG_detection)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    years = [str(y) for y in args.years]

    print(f"Workspace: {root}")
    print(f"Python:    {py}")
    print(f"Years:     {', '.join(years)}")
    print(f"CSV out:   {out_dir}")

    year_bar = (
        tqdm(
            total=len(years),
            desc="Years",
            unit="year",
            ascii=True,
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    try:
        for year in years:
            paths = build_year_paths(root, out_dir, year)
            run_year(root, py, paths, out_dir)
            if year_bar is not None:
                year_bar.update(1)
    finally:
        if year_bar is not None:
            year_bar.close()

    print("\nAll requested years finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
