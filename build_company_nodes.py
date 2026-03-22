"""
build_company_nodes.py
======================
Build a single node list (CSV) of companies that appear in extracted business
text for a given year: union of ``{year}_10k_business`` (or ``{year}_10K_business``)
and ``{year}_no_outgoing_edges``.

Each row: company name, CIK, ticker, GICS sector.

- **Name / CIK** are extracted from the filename (one row per parsed ``*_business.txt`` file).
- **Ticker** — Yahoo Finance search via **yfinance** using company name.
- **GICS sector** — Yahoo Finance via **yfinance** (``info["sector"]``). Yahoo’s
    sector labels for U.S. listings are GICS-aligned; install: ``pip install yfinance``.

Optional ``--enrichment`` CSV overrides any of the above per CIK.

CIK resolution policy: prefer issuer CIK from the SEC raw folder structure
(``SEC/{year}_10K_raw/<issuer_cik>/...``), then fall back to the first 10
digits of the accession segment in the filename (submitter CIK), normalized to
integer-string form.

Usage:
        pip install yfinance
        python build_company_nodes.py --year 2024
        python build_company_nodes.py --year 2023 --base-dir /path/to/data
        python build_company_nodes.py --year 2024 --enrichment ./cik_industry_2024.csv
        python build_company_nodes.py --year 2024 --no-fetch-tickers
        python build_company_nodes.py --year 2024 --no-fetch-classifications
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

from build_manifest import parse_filename

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_SCRIPT_DIR = Path(__file__).resolve().parent

NODE_FIELDS = [
    "company_name",
    "cik",
    "ticker",
    "gics_sector",
]

ACCESSION_IN_THIRD_SEGMENT_RE = re.compile(r"(\d{10})-\d{2}-\d{6}")


def _with_progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    """Wrap an iterable with tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="row")


def _norm_cik(val: str | None) -> str:
    if not val or not str(val).strip():
        return ""
    s = str(val).strip()
    if not s.isdigit():
        return ""
    return str(int(s))


def _cik_10(cik: str) -> str:
    return str(int(cik)).zfill(10)


def _name_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _cik_from_third_segment(stem: str) -> str:
    """Extract CIK from first 10 digits of accession in the third '__' segment."""
    parts = stem.split("__")
    if len(parts) < 3:
        return ""
    m = ACCESSION_IN_THIRD_SEGMENT_RE.search(parts[2])
    if not m:
        return ""
    return _norm_cik(m.group(1))


def _stem_without_business_suffix(stem: str) -> str:
    if stem.endswith("_business"):
        return stem[: -len("_business")]
    return stem


def _build_sec_raw_issuer_cik_lookup(base: Path, year: str) -> dict[str, str]:
    """
    Build a map from raw filing stem -> issuer CIK using SEC raw folder layout:
    SEC/{year}_10K_raw/<issuer_cik>/*.txt (or lowercase k variant).
    """
    lookup: dict[str, str] = {}
    roots = [
        (base / "SEC" / f"{year}_10K_raw").resolve(),
        (base / "SEC" / f"{year}_10k_raw").resolve(),
    ]

    for root in _dedupe_paths_by_inode(roots):
        if not root.is_dir():
            continue
        for cik_dir in root.iterdir():
            if not cik_dir.is_dir():
                continue
            issuer_cik = _norm_cik(cik_dir.name)
            if not issuer_cik:
                continue
            for txt in cik_dir.glob("*.txt"):
                stem_key = txt.stem.strip().lower()
                if not stem_key:
                    continue
                # Keep first-seen value; collisions are ignored deterministically.
                lookup.setdefault(stem_key, issuer_cik)

    return lookup


def _yfinance_ticker_from_company(company_name: str) -> str:
    if not company_name:
        return ""
    try:
        import yfinance as yf
    except ImportError:
        return ""
    try:
        quotes = (yf.Search(company_name, max_results=8).quotes or [])
    except Exception:
        return ""

    best_symbol = ""
    best_score = -1
    src_tokens = _name_tokens(company_name)

    for q in quotes:
        symbol = str(q.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        qtype = str(q.get("quoteType") or q.get("type") or "").lower()
        name = str(q.get("shortname") or q.get("longname") or "")
        score = 0
        if qtype in {"equity", "stock"}:
            score += 5
        if src_tokens and name:
            score += len(src_tokens & _name_tokens(name))
        if score > best_score:
            best_score = score
            best_symbol = symbol

    return best_symbol


def _yfinance_gics_sector(ticker: str) -> str:
    if not ticker:
        return ""
    try:
        import yfinance as yf
    except ImportError:
        return ""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return ""
    return str(info.get("sector") or info.get("sectorDisp") or "").strip()


def _dedupe_paths_by_inode(paths: list[Path]) -> list[Path]:
    """Drop paths that refer to the same file/dir (handles case-insensitive volumes)."""
    seen: set[tuple[int, int]] = set()
    out: list[Path] = []
    for p in paths:
        try:
            st = p.resolve().stat()
            key = (st.st_dev, st.st_ino)
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p.resolve())
    return out


def _resolve_business_dirs(base: Path, year: str) -> list[Path]:
    """Return existing directories for 10-K business + no_outgoing_edges."""
    candidates = [
        base / f"{year}_10k_business",
        base / f"{year}_10K_business",
        base / f"{year}_no_outgoing_edges",
    ]
    existing: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp.is_dir():
            existing.append(rp)
    return _dedupe_paths_by_inode(existing)


def _collect_business_txt(root: Path, recursive: bool) -> list[Path]:
    glob = root.rglob if recursive else root.glob
    return sorted({p for p in glob("*_business.txt") if p.is_file()})


_ENRICH_ALIASES = {
    "cik": ("cik", "CIK", "source_cik", "cik_raw"),
    "ticker": ("ticker", "TICKER", "symbol", "SYM"),
    "gics_sector": ("gics_sector", "gics", "GICS", "sector_gics"),
}

# Regex patterns to exclude funds, trusts, and NASDAQ entities from becoming nodes
_EXCLUDE_COMPANY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\btrustee\b"),
    re.compile(r"(?i)\btrust(?:\s+of\s+)?(?:account|fund|estate)?\b"),
    re.compile(r"(?i)\b(?:mutual|index|exchange-traded|closed-end|open-end|money market|hedge|pension|private)?(\s+)fund\b"),
    re.compile(r"(?i)\bfund\s+of\s+funds\b"),
    re.compile(r"(?i)nasdaq(?:\s*-?\s*(?:100|composite|composite|global|capital market))?(?:\s+index)?$"),
    re.compile(r"(?i)^nasdaq\b"),
    re.compile(r"(?i)\bindex\s+fund\b"),
    re.compile(r"(?i)\bclosed[\s-]?end\b"),
    re.compile(r"(?i)\blagged\s+trust\b"),
)


def _should_exclude_company(company_name: str) -> bool:
    """
    Return True if the company should be excluded from nodes.
    Filters out funds, trusts, and NASDAQ entities.
    """
    if not company_name:
        return False
    
    for pattern in _EXCLUDE_COMPANY_PATTERNS:
        if pattern.search(company_name):
            return True
    
    return False


def _enrichment_row_to_dict(header: list[str], row: list[str]) -> dict[str, str]:
    hmap = {h.strip(): i for i, h in enumerate(header)}
    out: dict[str, str] = {}
    for canon, aliases in _ENRICH_ALIASES.items():
        for a in aliases:
            if a in hmap and hmap[a] < len(row):
                v = row[hmap[a]].strip()
                if v:
                    out[canon] = v
                    break
    return out


def _load_enrichment(path: Path) -> dict[str, dict[str, str]]:
    """Map CIK -> optional ticker, naics_industry, gics_sector."""
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return out
        for row in _with_progress(reader, desc="Loading enrichment"):
            if not row:
                continue
            d = _enrichment_row_to_dict(header, row)
            cik = _norm_cik(d.get("cik", ""))
            if not cik:
                continue
            out[cik] = {k: v for k, v in d.items() if k != "cik" and v}
    return out


def _merge_enrichment(
    nodes: list[dict[str, str]],
    enrich: dict[str, dict[str, str]],
) -> None:
    for row in nodes:
        cik = row["cik"]
        if not cik:
            continue
        ex = enrich.get(cik)
        if ex:
            if ex.get("ticker"):
                row["ticker"] = ex["ticker"]
            if ex.get("gics_sector"):
                row["gics_sector"] = ex["gics_sector"]


def _fill_tickers_from_yfinance(nodes: list[dict[str, str]], sleep_s: float) -> int:
    filled = 0
    cache: dict[str, str] = {}
    for row in _with_progress(nodes, desc="Filling tickers (yfinance)", total=len(nodes)):
        if row["ticker"]:
            continue
        key = row["company_name"].strip().lower()
        if key in cache:
            tick = cache[key]
        else:
            tick = _yfinance_ticker_from_company(row["company_name"])
            cache[key] = tick
            if sleep_s > 0:
                time.sleep(sleep_s)
        if tick:
            row["ticker"] = tick
            filled += 1
    return filled


def _fill_classifications(
    nodes: list[dict[str, str]],
    *,
    enabled: bool,
    yfinance_sleep_s: float,
) -> int:
    """
    Fill gics_sector via yfinance for rows that already have ticker.
    Returns count: gics_filled.
    """
    if not enabled:
        return 0

    try:
        import yfinance  # noqa: F401

        have_yf = True
    except ImportError:
        have_yf = False
        print(
            "[warn] yfinance not installed; gics_sector will stay empty "
            "(pip install yfinance).",
            file=sys.stderr,
        )

    gics_n = 0

    for row in _with_progress(nodes, desc="Filling GICS sector", total=len(nodes)):
        cik = row["cik"]
        if not cik:
            continue
        if have_yf and not row["gics_sector"] and row["ticker"]:
            sec = _yfinance_gics_sector(row["ticker"])
            if sec:
                row["gics_sector"] = sec
                gics_n += 1
            if yfinance_sleep_s > 0:
                time.sleep(yfinance_sleep_s)

    return gics_n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a company node CSV from 10k_business + no_outgoing_edges folders."
    )
    p.add_argument("--year", required=True, help="Calendar year (e.g. 2024)")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory containing {year}_10k_business and {year}_no_outgoing_edges",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: {base-dir}/{year}_company_nodes.csv)",
    )
    p.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Scan subdirectories for *_business.txt",
    )
    p.add_argument(
        "--enrichment",
        type=Path,
        default=None,
        help="Optional CSV with CIK-keyed ticker / GICS columns",
    )
    p.add_argument(
        "--fetch-tickers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use yfinance search to fill missing tickers from company names (default: on)",
    )
    p.add_argument(
        "--fetch-classifications",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use yfinance to fill gics_sector from ticker (default: on)",
    )
    p.add_argument(
        "--yfinance-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between yfinance requests (default: 0.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    year = args.year.strip()
    if not re.fullmatch(r"\d{4}", year):
        sys.exit("[ERROR] --year must be a four-digit year")

    base = args.base_dir.resolve()
    out_path = args.output
    if out_path is None:
        out_path = base / f"{year}_company_nodes.csv"

    dirs = _resolve_business_dirs(base, year)
    if not dirs:
        tried = f"{year}_10k_business | {year}_10K_business | {year}_no_outgoing_edges"
        sys.exit(f"[ERROR] No such directories under {base} ({tried})")

    all_paths: list[Path] = []
    for d in dirs:
        all_paths.extend(_collect_business_txt(d, args.recursive))

    if not all_paths:
        sys.exit(
            f"[ERROR] No *_business.txt files under: "
            + ", ".join(str(d) for d in dirs)
        )

    if tqdm is None:
        print("[warn] tqdm not installed; progress bars disabled.", file=sys.stderr)

    raw_rows: list[dict[str, Any]] = []
    parse_failures: list[tuple[str, str]] = []
    sec_raw_lookup = _build_sec_raw_issuer_cik_lookup(base, year)
    cik_from_sec_raw_n = 0

    deduped_paths = sorted(_dedupe_paths_by_inode(all_paths), key=lambda p: p.as_posix().lower())
    for path in _with_progress(deduped_paths, desc="Parsing business files", total=len(deduped_paths)):
        stem = path.stem
        parsed = parse_filename(stem)
        raw_stem_key = _stem_without_business_suffix(stem).strip().lower()
        cik_from_raw = sec_raw_lookup.get(raw_stem_key, "")
        cik = cik_from_raw or _cik_from_third_segment(stem) or _norm_cik(parsed.get("submitter_cik") or parsed.get("cik_raw"))
        name = (parsed.get("source_company_name") or "").strip()
        if not cik or not name:
            parse_failures.append(
                (path.name, parsed.get("parse_notes") or "missing cik or company name")
            )
            continue
        if cik_from_raw:
            cik_from_sec_raw_n += 1
        raw_rows.append(
            {
                "company_name": name,
                "cik": cik,
            }
        )

    nodes: list[dict[str, str]] = []
    excluded_companies: list[str] = []
    for r in _with_progress(raw_rows, desc="Building node rows", total=len(raw_rows)):
        name = r["company_name"]
        if _should_exclude_company(name):
            excluded_companies.append(name)
            continue
        nodes.append(
            {
                "company_name": name,
                "cik": r["cik"],
                "ticker": "",
                "gics_sector": "",
            }
        )

    enrich_map: dict[str, dict[str, str]] = {}
    if args.enrichment:
        ep = args.enrichment.resolve()
        if not ep.is_file():
            sys.exit(f"[ERROR] --enrichment not found: {ep}")
        enrich_map = _load_enrichment(ep)

    _merge_enrichment(nodes, enrich_map)

    yf_ticker_n = 0
    if args.fetch_tickers:
        yf_ticker_n = _fill_tickers_from_yfinance(nodes, sleep_s=args.yfinance_sleep)

    gics_n = 0
    if args.fetch_classifications:
        gics_n = _fill_classifications(
            nodes,
            enabled=True,
            yfinance_sleep_s=args.yfinance_sleep,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NODE_FIELDS)
        w.writeheader()
        w.writerows(nodes)

    issues_path = out_path.with_suffix(".parse_issues.txt")
    if parse_failures:
        lines = [
            f"Files skipped (could not parse CIK + company name): {len(parse_failures)}",
            "",
        ]
        for fn, note in parse_failures:
            lines.append(f"{fn}\n  {note}\n")
        issues_path.write_text("\n".join(lines), encoding="utf-8")
    elif issues_path.is_file():
        issues_path.unlink()
    
    excluded_path = out_path.with_suffix(".excluded_companies.txt")
    if excluded_companies:
        lines = [
            f"Companies excluded (funds/trusts/NASDAQ): {len(excluded_companies)}",
            "",
        ] + [name for name in sorted(set(excluded_companies))]
        excluded_path.write_text("\n".join(lines), encoding="utf-8")
    elif excluded_path.is_file():
        excluded_path.unlink()

    print(f"Directories scanned: {len(dirs)}")
    for d in dirs:
        print(f"  • {d}")
    print(f"Unique *_business.txt paths: {len(set(all_paths)):,}")
    print(f"CIK from SEC raw issuer folders: {cik_from_sec_raw_n:,}")
    print(f"Raw company rows extracted:      {len(raw_rows):,}")
    print(f"Companies excluded (funds/trusts/NASDAQ): {len(excluded_companies):,}")
    print(f"Node rows written:           {len(nodes):,}")
    print(f"Unique CIKs in rows:         {len({n['cik'] for n in nodes if n['cik']}):,}")
    print(f"Rows missing CIK:            {sum(1 for n in nodes if not n['cik']):,}")
    print(f"Wrote: {out_path}")
    if parse_failures:
        print(f"Parse issues: {issues_path}")
    if excluded_companies:
        print(f"Excluded companies: {excluded_path}")
    if args.fetch_tickers:
        filled = sum(1 for n in nodes if n["ticker"])
        print(
            f"Tickers filled (yfinance): {filled:,} / {len(nodes):,} "
            f"(new this run: {yf_ticker_n:,})"
        )
    if args.fetch_classifications:
        print(f"Classifications — GICS sector (yfinance): {gics_n:,}")


if __name__ == "__main__":
    main()
