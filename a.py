"""
COMPREHENSIVE 10-K PROCESSING PIPELINE
Processes raw SEC filings, cleans HTML, extracts business/competition sections,
 2024 data.

Features:
  • HTML cleaning: strips XBRL, metadata, non-text blocks
  • Business/competition section extraction
  • Audit categorization (no business, competition header only, no competition term)
  • Isolated folder organization
  • JSON and CSV output

Run:
    python a.py
"""

from __future__ import annotations
import html
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, FeatureNotFound
from tqdm import tqdm

# ── CONFIGURATION ────────────────────────────────────────────────────────────
PATH = "C:\\JuneKim0712\\Code\\daeun\\SEC\\"

INPUT_DIRS = {
    "2024": Path(PATH + "2024_10K_raw"),
}

OUTPUT_DIRS = {
    "2024_cleaned": Path("2024_10K_cleaned"),
    "2024_business": Path("2024_10K_business"),
    "2024_isolated": Path("2024_10K_isolated"),
    "2024_no_outgoing_edges": Path("2024_no_outgoing_edges"),
}

VALID_SUFFIXES = {".txt", ".html", ".htm", ".xml"}

# ── REGEX PATTERNS ───────────────────────────────────────────────────────────
PRIMARY_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.I | re.S)
DOC_TYPE_RE = re.compile(r"<TYPE>\s*([^\n\r<]+)", re.I)
TEXT_BLOCK_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.I | re.S)

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

COMPETITION_TERM_RE = re.compile(
    r"\b(competition|competitor|competitors|competitive|compete|competes|competing)\b",
    re.I,
)

REMOVABLE_HTML_TAGS = {
    "script", "style", "noscript", "head", "meta", "link", "svg", "img",
    "object", "iframe", "footer", "nav", "form",
}

SEC_METADATA_LINE_PATTERNS = [
    r"^<PAGE>$", r"^<TYPE>.*$", r"^<SEQUENCE>.*$", r"^<FILENAME>.*$",
    r"^<DESCRIPTION>.*$", r"^accession number\b.*$", r"^conformed period of report\b.*$",
    r"^accepted\b.*$", r"^public document count\b.*$", r"^central index key\b.*$",
    r"^standard industrial classification\b.*$", r"^irs number\b.*$",
    r"^state of incorporation\b.*$", r"^fiscal year end\b.*$",
    r"^commission file number\b.*$", r"^company conformed name\b.*$",
]

TOC_LINE_PATTERNS = [
    r"^table of contents$", r"^index to financial statements$",
    r"^part\s+[ivx]+\s+\.{2,}\s*\d+\s*$", r"^item\s+\d+[a-z]?(?:\.[^.]*)?\.{2,}\s*\d+\s*$",
]

XBRL_DROP_BLOCKS = [
    r"(?is)<ix:header\b.*?</ix:header>", r"(?is)<ix:hidden\b.*?</ix:hidden>",
    r"(?is)<ix:references\b.*?</ix:references>", r"(?is)<ix:resources\b.*?</ix:resources>",
    r"(?is)<xbrli:context\b.*?</xbrli:context>", r"(?is)<xbrli:unit\b.*?</xbrli:unit>",
    r"(?is)<xbrli:entity\b.*?</xbrli:entity>", r"(?is)<xbrli:period\b.*?</xbrli:period>",
    r"(?is)<xbrldi:explicitmember\b.*?</xbrldi:explicitmember>",
    r"(?is)<xbrldi:typedmember\b.*?</xbrldi:typedmember>",
    r"(?is)<link:schemaref\b[^>]*?/?>", r"(?is)<link:linkbaseref\b[^>]*?/?>",
]

INLINE_XBRL_UNWRAP_TAGS = {"ix:nonnumeric", "ix:nonfraction", "ix:fraction", "ix:continuation"}

TAXONOMY_TOKEN_RE = re.compile(
    r"(?i)\b(?:us-gaap|dei|xbrli|xbrldi|ix|ixt|srt|stpr|iso4217|country|ecd):[A-Za-z0-9_.#/-]+\b"
)
MEMBER_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.-]*Member\b")
URL_TOKEN_RE = re.compile(r"https?://\S+")

PREAMBLE_START_RES = [
    re.compile(r"(?is)\bform\s+10-k\b"),
    re.compile(r"(?is)\bannual\s+report\s+pursuant\s+to\s+section\s+13\s+or\s+15\s*\(d\)\b"),
    re.compile(r"(?is)\bpart\s+i\b"),
]

def choose_html_parser() -> str:
    try:
        BeautifulSoup("<html></html>", "lxml")
        return "lxml"
    except FeatureNotFound:
        return "html.parser"

HTML_PARSER = choose_html_parser()

# ── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def read_text_robust(path: Path) -> str:
    """Read file with multiple encoding fallbacks."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_primary_10k_document(raw: str) -> str:
    """Extract the main 10-K document from SEC filing."""
    docs = PRIMARY_DOC_RE.findall(raw)
    if not docs:
        return raw

    candidates = []
    for doc in docs:
        m = DOC_TYPE_RE.search(doc)
        doc_type = m.group(1).strip().upper() if m else ""
        if doc_type.startswith("10-K") or doc_type in {"10K", "10-K405", "10-KT", "10KSB"}:
            candidates.append(doc)

    chosen = max(candidates, key=len) if candidates else max(docs, key=len)
    text_match = TEXT_BLOCK_RE.search(chosen)
    return text_match.group(1) if text_match else chosen


def strip_non_text_blocks(raw: str) -> str:
    """Remove PDF, Excel, ZIP blocks."""
    patterns = [r"(?is)<PDF>.*?</PDF>", r"(?is)<EXCEL>.*?</EXCEL>", r"(?is)<ZIP>.*?</ZIP>"]
    out = raw
    for pat in patterns:
        out = re.sub(pat, " ", out)
    return out


def strip_xbrl_support_blocks(raw: str) -> str:
    """Remove XBRL metadata blocks."""
    out = raw
    for pat in XBRL_DROP_BLOCKS:
        out = re.sub(pat, " ", out)
    return out


def is_hidden_tag(tag) -> bool:
    """Check if HTML tag is hidden."""
    attrs = getattr(tag, "attrs", {}) or {}
    style = str(attrs.get("style", "")).replace(" ", "").lower()
    hidden_attr = str(attrs.get("hidden", "")).lower()
    aria_hidden = str(attrs.get("aria-hidden", "")).lower()
    return "display:none" in style or "visibility:hidden" in style or hidden_attr in {"hidden", "true"} or aria_hidden == "true"


def html_to_text(raw: str) -> str:
    """Convert HTML to plain text, removing tags and hidden content."""
    raw = strip_non_text_blocks(raw)
    raw = strip_xbrl_support_blocks(raw)
    soup = BeautifulSoup(raw, HTML_PARSER)

    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()
        if name in REMOVABLE_HTML_TAGS or is_hidden_tag(tag):
            tag.decompose()
        elif name in INLINE_XBRL_UNWRAP_TAGS:
            tag.unwrap()
        elif name.startswith(("xbrli:", "xbrldi:", "link:")):
            tag.decompose()
        elif name.startswith("ix:"):
            tag.unwrap()

    for tag in soup.find_all(["br", "p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return soup.get_text(separator=" ")


def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize unicode and whitespace."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u00ad", "").replace("\x0c", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?i)(?<!\n)\b(PART\s+[IVX]+)\b", r"\n\1", text)
    text = re.sub(r"(?i)(?<!\n)\b(ITEM\s+\d+[A-Z]?)\b", r"\n\1", text)
    return text.strip()


def is_mostly_xbrl_noise(line: str) -> bool:
    """Check if line is mostly XBRL/taxonomy noise."""
    s = line.strip()
    if not s:
        return False
    tokens = [tok.strip(".,;:()[]{}") for tok in s.split()]
    if len(tokens) < 6:
        return False
    junk = sum(1 for tok in tokens if TAXONOMY_TOKEN_RE.fullmatch(tok) or MEMBER_TOKEN_RE.fullmatch(tok) or URL_TOKEN_RE.fullmatch(tok))
    return (junk / max(len(tokens), 1)) >= 0.55


def is_junk_line(line: str) -> bool:
    """Check if line is metadata or junk."""
    s = line.strip()
    if not s:
        return False
    for pat in SEC_METADATA_LINE_PATTERNS + TOC_LINE_PATTERNS:
        if re.match(pat, s, flags=re.I):
            return True
    if re.match(r"^(page\s+)?\d{1,4}$", s, flags=re.I) or is_mostly_xbrl_noise(s):
        return True
    return False


def is_heading(line: str) -> bool:
    """Check if line is a section heading."""
    s = line.strip()
    return bool(re.match(r"(?i)^part\s+[ivx]+\b", s) or re.match(r"(?i)^item\s+\d+[a-z]?\b", s))


def rebuild_paragraphs(text: str) -> str:
    """Reconstruct paragraphs from line-by-line text."""
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line or is_junk_line(line):
            lines.append("" if not line else None)
        else:
            lines.append(line)

    paragraphs, buf = [], []

    def flush():
        nonlocal buf
        if buf:
            para = " ".join(buf).strip()
            if para:
                paragraphs.append(para)
            buf = []

    for line in lines:
        if line is None:
            continue
        if not line:
            flush()
        elif is_heading(line):
            flush()
            paragraphs.append(line.upper())
        else:
            buf.append(line)
            if line.endswith((".", "!", "?", ":", ";")):
                flush()

    flush()
    return "\n\n".join(paragraphs).strip()


def trim_leading_preamble(text: str) -> str:
    """Remove leading preamble before main content."""
    matches = [m.start() for pat in PREAMBLE_START_RES if (m := pat.search(text)) and m.start() > 1000]
    return text[min(matches):].strip() if matches else text


def clean_filing(raw_text: str) -> str:
    """Complete cleaning pipeline."""
    raw_text = extract_primary_10k_document(raw_text)
    text = html_to_text(raw_text)
    text = normalize_unicode_and_whitespace(text)
    text = rebuild_paragraphs(text)
    text = trim_leading_preamble(text)
    return text


def extract_section_by_patterns(
    text: str,
    start_patterns: list[tuple[str, re.Pattern]],
    end_patterns: list[re.Pattern],
    min_len: int,
    max_len: int = 500_000,
) -> Optional[str]:
    """Extract section matching start/end patterns."""
    if not any(start_re.search(text) for _, start_re in start_patterns):
        return None

    best: Optional[str] = None
    for _label, start_re in start_patterns:
        candidates = []
        for m in start_re.finditer(text):
            start = m.start()
            surrounding = text[max(0, start - 5): m.end() + 20]
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


def extract_business_section(cleaned_text: str) -> Optional[str]:
    return extract_section_by_patterns(cleaned_text, BUSINESS_START_PATTERNS, BUSINESS_END_RES, min_len=500)


def extract_competition_section(cleaned_text: str) -> Optional[str]:
    return extract_section_by_patterns(cleaned_text, COMPETITION_START_PATTERNS, COMPETITION_END_RES, min_len=300)


def extract_business_header_only(cleaned_text: str) -> Optional[str]:
    return extract_section_by_patterns(cleaned_text, BUSINESS_START_PATTERNS, BUSINESS_END_RES, min_len=20, max_len=499)


# ── PROCESSING LOGIC ────────────────────────────────────────────────────

def process_file(src_path: Path, cleaned_dir: Path, business_dir: Path, isolated_dir: Path, no_outgoing_dir: Path) -> dict:
    """Process a single filing through complete pipeline."""
    raw = read_text_robust(src_path)
    cleaned = clean_filing(raw)
    has_business_header = any(start_re.search(cleaned) for _, start_re in BUSINESS_START_PATTERNS)
    has_competition_header = any(start_re.search(cleaned) for _, start_re in COMPETITION_START_PATTERNS)
    has_competition_term_anywhere = bool(COMPETITION_TERM_RE.search(cleaned))

    section_text: Optional[str] = None
    section_source: Optional[str] = None
    audit_category = None
    audit_reason = None
    isolate = False

    # Header-driven extraction: business first, then competition fallback.
    business = extract_business_section(cleaned) if has_business_header else None
    if business is not None:
        section_text = business
        section_source = "business"
    else:
        competition = extract_competition_section(cleaned) if has_competition_header else None
        if competition is not None:
            section_text = competition
            section_source = "competition"

    # Isolation rules after attempting both header-based extractions.
    if section_text is None:
        if not has_business_header and not has_competition_header:
            isolate = True
            if has_competition_term_anywhere:
                audit_category = "no_business_header_has_competition_term"
                audit_reason = "No business/competition headers found, but competition terms appear in text."
            else:
                audit_category = "missing_both_no_competition_term"
                audit_reason = "No business/competition headers and no competition terms found."
        elif has_business_header or has_competition_header:
            isolate = True
            audit_category = "business_header_only"
            audit_reason = "Business or competition header found, but no extractable section content."

    has_competition_term_in_business = bool(COMPETITION_TERM_RE.search(section_text)) if section_text else False

    cleaned_name = src_path.stem + ".txt"
    section_name = src_path.stem + "_business.txt"
    cleaned_path = cleaned_dir / cleaned_name
    business_path = business_dir / section_name

    if isolate:
        cat_dir = isolated_dir / audit_category
        cat_dir.mkdir(parents=True, exist_ok=True)
        isolated_cleaned_path = cat_dir / cleaned_name
        isolated_cleaned_path.write_text(cleaned, encoding="utf-8")
        if section_text:
            isolated_section_path = cat_dir / section_name
            isolated_section_path.write_text(section_text, encoding="utf-8")
        if cleaned_path.exists():
            cleaned_path.unlink()
        if business_path.exists():
            business_path.unlink()
    elif section_text and section_source == "business" and not has_competition_header and not has_competition_term_in_business:
        # Files with no outgoing edges: business section + NO competition header + NO competition terms in business
        cleaned_path.write_text(cleaned, encoding="utf-8")
        no_outgoing_path = no_outgoing_dir / section_name
        no_outgoing_path.write_text(section_text, encoding="utf-8")
        if business_path.exists():
            business_path.unlink()
    else:
        cleaned_path.write_text(cleaned, encoding="utf-8")
        if section_text:
            business_path.write_text(section_text, encoding="utf-8")
        elif business_path.exists():
            business_path.unlink()

    return {
        "source_file": str(src_path),
        "cleaned_file": str(cleaned_path) if not isolate else None,
        "business_file": str(business_path) if (section_text and not isolate and not (section_source == "business" and not has_competition_header and not has_competition_term_in_business)) else None,
        "no_outgoing_edges_file": str(no_outgoing_dir / section_name) if (section_text and section_source == "business" and not isolate and not has_competition_header and not has_competition_term_in_business) else None,
        "cleaned_chars": len(cleaned),
        "has_business_header": has_business_header,
        "has_competition_header": has_competition_header,
        "has_business_section": section_text is not None,
        "business_chars": len(section_text) if section_text else 0,
        "has_competition_term_anywhere": has_competition_term_anywhere,
        "has_competition_term_in_business": has_competition_term_in_business,
        "section_source": section_source,
        "audit_category": audit_category,
        "audit_reason": audit_reason,
        "is_isolated": isolate,
    }


def iter_input_files(folder: Path):
    """Iterate through all valid input files."""
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def write_audit_files(year: str, manifest: list[dict]) -> None:
    """Write audit JSON and CSV files."""
    out_json = Path(f"{year}_10k_audit.json")
    out_csv = Path(f"{year}_10k_audit.csv")

    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred_fields = [
        "source_file", "cleaned_file", "business_file", "no_outgoing_edges_file",
        "cleaned_chars", "has_business_section", "business_chars",
        "has_competition_term_anywhere", "has_competition_term_in_business",
        "section_source", "audit_category", "audit_reason", "is_isolated",
    ]

    seen = set(preferred_fields)
    extra_fields = [k for row in manifest for k in row.keys() if k not in seen and (seen.add(k) or True)]

    fieldnames = preferred_fields + extra_fields
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"  Wrote: {out_json.name}")
    print(f"  Wrote: {out_csv.name}")


def write_isolated_audit(year: str, manifest: list[dict]) -> None:
    """Write isolated/categorized files to separate audit."""
    isolated = [row for row in manifest if row.get("is_isolated")]
    out_json = Path(f"{year}_10k_isolated_audit.json")
    out_csv = Path(f"{year}_10k_isolated_audit.csv")

    out_json.write_text(json.dumps(isolated, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "audit_category", "audit_reason", "source_file", "cleaned_chars",
        "has_business_section", "has_competition_term_anywhere", "section_source", "is_isolated",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(isolated)

    print(f"  Wrote: {out_json.name}")
    print(f"  Wrote: {out_csv.name}")


def write_category_lists(year: str, manifest: list[dict]) -> None:
    """Write separate lists for different audit categories."""
    categories = {}
    for row in manifest:
        if row.get("is_isolated"):
            cat = row.get("audit_category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row)

    for cat_name, rows in categories.items():
        out_file = Path(f"{year}_{cat_name}_list.json")
        out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Wrote: {out_file.name} ({len(rows)} entries)")


def process_year(year: str, input_dir: Path, cleaned_dir: Path, business_dir: Path, isolated_dir: Path, no_outgoing_dir: Path, limit: int = 1000000) -> None:
    """Process all files for a given year."""
    if not input_dir.exists():
        print(f"[{year}] SKIP: input folder not found: {input_dir}")
        return

    for d in [cleaned_dir, business_dir, isolated_dir, no_outgoing_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(input_dir))[:limit]
    manifest = []

    counts = {
        "business": 0,
        "competition": 0,
        "no_outgoing_edges": 0,
        "missing_both_no_competition_term": 0,
        "competition_header_only_no_term": 0,
        "no_business_header_has_competition_term": 0,
        "business_header_only": 0,
        "errors": 0,
    }

    pbar = tqdm(files, desc=f"[{year}] Processing", unit="file")
    for src_path in pbar:
        try:
            meta = process_file(src_path, cleaned_dir, business_dir, isolated_dir, no_outgoing_dir)
            manifest.append(meta)

            if meta.get("no_outgoing_edges_file"):
                counts["no_outgoing_edges"] += 1
            elif meta.get("section_source") == "business":
                counts["business"] += 1
            elif meta.get("section_source") == "competition":
                counts["competition"] += 1

            if meta.get("is_isolated"):
                cat = meta.get("audit_category", "unknown")
                counts[cat] = counts.get(cat, 0) + 1

            pbar.set_postfix(
                business=counts["business"],
                no_outgoing=counts["no_outgoing_edges"],
                missing=counts["missing_both_no_competition_term"],
                comp_only=counts["competition_header_only_no_term"],
            )
        except Exception as e:
            manifest.append({"source_file": str(src_path), "error": str(e)})
            counts["errors"] += 1

    write_audit_files(year, manifest)
    write_isolated_audit(year, manifest)
    write_category_lists(year, manifest)

    print(f"\n[{year}] ── PROCESSING SUMMARY ─────────────────────────────")
    print(f"[{year}] Total files processed        : {len(files):,}")
    print(f"[{year}] Business sections (outgoing): {counts['business']:,}")
    print(f"[{year}] No outgoing edges (has comp): {counts['no_outgoing_edges']:,}")
    print(f"[{year}] Competition fallback        : {counts['competition']:,}")
    print(f"[{year}] Missing both+no comp term   : {counts['missing_both_no_competition_term']:,}")
    print(f"[{year}] Competition header only     : {counts['competition_header_only_no_term']:,}")
    print(f"[{year}] No business header          : {counts['no_business_header_has_competition_term']:,}")
    print(f"[{year}] Business header only        : {counts['business_header_only']:,}")


def main():
    """Process 2024 data."""
    print("Starting comprehensive 10-K processing pipeline for 2024...\n")
    
    year = "2024"
    process_year(
        year,
        INPUT_DIRS[year],
        OUTPUT_DIRS[f"{year}_cleaned"],
        OUTPUT_DIRS[f"{year}_business"],
        OUTPUT_DIRS[f"{year}_isolated"],
        OUTPUT_DIRS[f"{year}_no_outgoing_edges"],
        limit=500000,
    )

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()