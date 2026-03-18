# pip install beautifulsoup4 lxml
from __future__ import annotations

import html
import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, FeatureNotFound
from tqdm import tqdm

PATH = "C:\\JuneKim0712\\Code\\daeun\\SEC\\"

INPUT_DIRS = {
    "2023": Path(PATH + "2023_10K_raw"),
    "2024": Path(PATH + "2024_10K_raw"),
}

OUTPUT_DIRS = {
    "2023_cleaned": Path("2023_10K_cleaned"),
    "2023_business": Path("2023_10K_business"),
    "2023_isolated": Path("2023_10K_isolated"),
    "2024_cleaned": Path("2024_10K_cleaned"),
    "2024_business": Path("2024_10K_business"),
    "2024_isolated": Path("2024_10K_isolated"),
}

VALID_SUFFIXES = {".txt", ".html", ".htm", ".xml"}

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
    "script",
    "style",
    "noscript",
    "head",
    "meta",
    "link",
    "svg",
    "img",
    "object",
    "iframe",
    "footer",
    "nav",
    "form",
}

SEC_METADATA_LINE_PATTERNS = [
    r"^<PAGE>$",
    r"^<TYPE>.*$",
    r"^<SEQUENCE>.*$",
    r"^<FILENAME>.*$",
    r"^<DESCRIPTION>.*$",
    r"^accession number\b.*$",
    r"^conformed period of report\b.*$",
    r"^accepted\b.*$",
    r"^public document count\b.*$",
    r"^central index key\b.*$",
    r"^standard industrial classification\b.*$",
    r"^irs number\b.*$",
    r"^state of incorporation\b.*$",
    r"^fiscal year end\b.*$",
    r"^commission file number\b.*$",
    r"^company conformed name\b.*$",
]

TOC_LINE_PATTERNS = [
    r"^table of contents$",
    r"^index to financial statements$",
    r"^part\s+[ivx]+\s+\.{2,}\s*\d+\s*$",
    r"^item\s+\d+[a-z]?(?:\.[^.]*)?\.{2,}\s*\d+\s*$",
]

XBRL_DROP_BLOCKS = [
    r"(?is)<ix:header\b.*?</ix:header>",
    r"(?is)<ix:hidden\b.*?</ix:hidden>",
    r"(?is)<ix:references\b.*?</ix:references>",
    r"(?is)<ix:resources\b.*?</ix:resources>",
    r"(?is)<xbrli:context\b.*?</xbrli:context>",
    r"(?is)<xbrli:unit\b.*?</xbrli:unit>",
    r"(?is)<xbrli:entity\b.*?</xbrli:entity>",
    r"(?is)<xbrli:period\b.*?</xbrli:period>",
    r"(?is)<xbrldi:explicitmember\b.*?</xbrldi:explicitmember>",
    r"(?is)<xbrldi:typedmember\b.*?</xbrldi:typedmember>",
    r"(?is)<link:schemaref\b[^>]*?/?>",
    r"(?is)<link:linkbaseref\b[^>]*?/?>",
]

INLINE_XBRL_UNWRAP_TAGS = {
    "ix:nonnumeric",
    "ix:nonfraction",
    "ix:fraction",
    "ix:continuation",
}

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


def read_text_robust(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def choose_html_parser() -> str:
    try:
        BeautifulSoup("<html></html>", "lxml")
        return "lxml"
    except FeatureNotFound:
        return "html.parser"


HTML_PARSER = choose_html_parser()


def extract_primary_10k_document(raw: str) -> str:
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
    if text_match:
        return text_match.group(1)

    return chosen


def strip_non_text_blocks(raw: str) -> str:
    patterns = [
        r"(?is)<PDF>.*?</PDF>",
        r"(?is)<EXCEL>.*?</EXCEL>",
        r"(?is)<ZIP>.*?</ZIP>",
    ]
    out = raw
    for pat in patterns:
        out = re.sub(pat, " ", out)
    return out


def strip_xbrl_support_blocks(raw: str) -> str:
    out = raw
    for pat in XBRL_DROP_BLOCKS:
        out = re.sub(pat, " ", out)
    return out


def is_hidden_tag(tag) -> bool:
    attrs = getattr(tag, "attrs", {}) or {}
    style = str(attrs.get("style", "")).replace(" ", "").lower()
    hidden_attr = str(attrs.get("hidden", "")).lower()
    aria_hidden = str(attrs.get("aria-hidden", "")).lower()

    return (
        "display:none" in style
        or "visibility:hidden" in style
        or hidden_attr in {"hidden", "true"}
        or aria_hidden == "true"
    )


def html_to_text(raw: str) -> str:
    raw = strip_non_text_blocks(raw)
    raw = strip_xbrl_support_blocks(raw)
    soup = BeautifulSoup(raw, HTML_PARSER)

    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()

        if name in REMOVABLE_HTML_TAGS:
            tag.decompose()
            continue

        if is_hidden_tag(tag):
            tag.decompose()
            continue

        if name in INLINE_XBRL_UNWRAP_TAGS:
            tag.unwrap()
            continue

        if name.startswith("xbrli:") or name.startswith("xbrldi:") or name.startswith("link:"):
            tag.decompose()
            continue

        if name.startswith("ix:"):
            tag.unwrap()
            continue

        if name in {"font", "span"}:
            continue

    for tag in soup.find_all(
        ["br", "p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return soup.get_text(separator=" ")


def normalize_unicode_and_whitespace(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u00ad", "")
    text = text.replace("\x0c", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"(?i)(?<!\n)\b(PART\s+[IVX]+)\b", r"\n\1", text)
    text = re.sub(r"(?i)(?<!\n)\b(ITEM\s+\d+[A-Z]?)\b", r"\n\1", text)

    return text.strip()


def is_mostly_xbrl_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    tokens = [tok.strip(".,;:()[]{}") for tok in s.split()]
    if len(tokens) < 6:
        return False

    junk = 0
    for tok in tokens:
        if TAXONOMY_TOKEN_RE.fullmatch(tok) or MEMBER_TOKEN_RE.fullmatch(tok) or URL_TOKEN_RE.fullmatch(tok):
            junk += 1

    return (junk / max(len(tokens), 1)) >= 0.55


def is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    for pat in SEC_METADATA_LINE_PATTERNS:
        if re.match(pat, s, flags=re.I):
            return True

    for pat in TOC_LINE_PATTERNS:
        if re.match(pat, s, flags=re.I):
            return True

    if re.match(r"^(page\s+)?\d{1,4}$", s, flags=re.I):
        return True

    if is_mostly_xbrl_noise(s):
        return True

    return False


def is_heading(line: str) -> bool:
    s = line.strip()
    return bool(
        re.match(r"(?i)^part\s+[ivx]+\b", s) or
        re.match(r"(?i)^item\s+\d+[a-z]?\b", s)
    )


def rebuild_paragraphs(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        if is_junk_line(line):
            continue
        lines.append(line)

    paragraphs = []
    buf = []

    def flush():
        nonlocal buf
        if buf:
            para = " ".join(buf).strip()
            if para:
                paragraphs.append(para)
            buf = []

    for line in lines:
        if not line:
            flush()
            continue

        if is_heading(line):
            flush()
            paragraphs.append(line.upper())
            continue

        buf.append(line)

        if line.endswith((".", "!", "?", ":", ";")):
            flush()

    flush()

    text = "\n\n".join(paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_leading_preamble(text: str) -> str:
    matches = []
    for pat in PREAMBLE_START_RES:
        m = pat.search(text)
        if m and m.start() > 1000:
            matches.append(m.start())

    if not matches:
        return text

    return text[min(matches):].strip()


def clean_filing(raw_text: str) -> str:
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
    return extract_section_by_patterns(
        cleaned_text,
        BUSINESS_START_PATTERNS,
        BUSINESS_END_RES,
        min_len=500,
    )


def extract_competition_section(cleaned_text: str) -> Optional[str]:
    return extract_section_by_patterns(
        cleaned_text,
        COMPETITION_START_PATTERNS,
        COMPETITION_END_RES,
        min_len=300,
    )


def extract_business_header_only(cleaned_text: str) -> Optional[str]:
    return extract_section_by_patterns(
        cleaned_text,
        BUSINESS_START_PATTERNS,
        BUSINESS_END_RES,
        min_len=20,
        max_len=499,
    )


def process_file(src_path: Path, cleaned_dir: Path, business_dir: Path, isolated_dir: Path) -> dict:
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

    # Audit-first fast triage: immediately isolate filings with no section signals.
    if (not has_business_header) and (not has_competition_header) and (not has_competition_term_anywhere):
        isolate = True
        audit_category = "missing_both_no_competition_term"
        audit_reason = "No business/competition section extracted and no competition term found anywhere in filing."
    else:
        # Only collect section snippets after a filing passes fast audit triage.
        business = extract_business_section(cleaned)
        if business is not None:
            section_text = business
            section_source = "business"
        else:
            competition = extract_competition_section(cleaned)
            if competition is not None:
                section_text = competition
                section_source = "competition"
            elif has_business_header:
                isolate = True
                audit_category = "business_header_only"
                audit_reason = "Business header detected but section content is too short; likely over-cleaned in prior step."

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
    else:
        cleaned_path.write_text(cleaned, encoding="utf-8")
        if section_text:
            business_path.write_text(section_text, encoding="utf-8")
        elif business_path.exists():
            business_path.unlink()

    return {
        "source_file": str(src_path),
        "cleaned_file": str(cleaned_path) if not isolate else None,
        "business_file": str(business_path) if (section_text and not isolate) else None,
        "cleaned_chars": len(cleaned),
        "has_business_header": has_business_header,
        "has_competition_header": has_competition_header,
        "has_business_section": section_text is not None,
        "business_chars": len(section_text) if section_text else 0,
        "has_competition_term_anywhere": has_competition_term_anywhere,
        "has_competition_term_in_business": bool(COMPETITION_TERM_RE.search(section_text)) if section_text else False,
        "recheck_section_source": section_source,
        "audit_category": audit_category,
        "audit_reason": audit_reason,
        "is_isolated": isolate,
    }


def iter_input_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def write_year_audit(year: str, manifest: list[dict]) -> None:
    out_json = Path(f"{year} 10k audit.json")
    out_csv = Path(f"{year} 10k audit.csv")

    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred_fields = [
        "source_file",
        "cleaned_file",
        "business_file",
        "cleaned_chars",
        "has_business_section",
        "business_chars",
        "has_competition_term_anywhere",
        "has_competition_term_in_business",
        "recheck_section_source",
        "audit_category",
        "audit_reason",
        "is_isolated",
        "error",
    ]

    seen = set(preferred_fields)
    extra_fields = []
    for row in manifest:
        for key in row.keys():
            if key not in seen:
                extra_fields.append(key)
                seen.add(key)

    fieldnames = preferred_fields + extra_fields
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def write_isolated_audit(year: str, manifest: list[dict]) -> None:
    isolated = [row for row in manifest if row.get("is_isolated")]
    out_json = Path(f"{year} 10k isolated audit.json")
    out_csv = Path(f"{year} 10k isolated audit.csv")

    out_json.write_text(json.dumps(isolated, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "audit_category",
        "audit_reason",
        "source_file",
        "cleaned_chars",
        "has_business_section",
        "has_competition_term_anywhere",
        "recheck_section_source",
        "is_isolated",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(isolated)


def process_year(
    year: str,
    input_dir: Path,
    cleaned_dir: Path,
    business_dir: Path,
    isolated_dir: Path,
    limit: int = 10,
) -> None:
    if not input_dir.exists():
        print(f"[{year}] SKIP: input folder not found: {input_dir}")
        return

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    business_dir.mkdir(parents=True, exist_ok=True)
    isolated_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(input_dir))[:limit]
    manifest = []

    counts = {
        "business": 0,
        "competition": 0,
        "business_header_only": 0,
        "missing_both_no_competition_term": 0,
        "errors": 0,
    }

    pbar = tqdm(files, desc=f"[{year}] Processing", unit="file")
    for idx, src_path in enumerate(pbar, start=1):
        try:
            meta = process_file(src_path, cleaned_dir, business_dir, isolated_dir)
            manifest.append(meta)

            source = meta.get("recheck_section_source")
            if source == "business":
                counts["business"] += 1
            elif source == "competition":
                counts["competition"] += 1
            elif source == "business_header_only":
                counts["business_header_only"] += 1

            if meta.get("audit_category") == "missing_both_no_competition_term":
                counts["missing_both_no_competition_term"] += 1

            pbar.set_postfix(
                business=counts["business"],
                competition=counts["competition"],
                header_only=counts["business_header_only"],
                missing_no_comp=counts["missing_both_no_competition_term"],
                errors=counts["errors"],
            )
        except Exception as e:
            manifest.append({"source_file": str(src_path), "error": str(e)})
            counts["errors"] += 1
            print(f"[{year}] {idx}/{len(files)} ERROR: {src_path.name} | {e}")

    write_year_audit(year, manifest)
    write_isolated_audit(year, manifest)

    print(f"\n[{year}] ── AUDIT SUMMARY ─────────────────────────")
    print(f"[{year}] total files                     : {len(files):,}")
    print(f"[{year}] business sections              : {counts['business']:,}")
    print(f"[{year}] competition fallback           : {counts['competition']:,}")
    print(f"[{year}] business header-only (isolated): {counts['business_header_only']:,}")
    print(f"[{year}] missing both + no comp term    : {counts['missing_both_no_competition_term']:,}")
    print(f"[{year}] errors                         : {counts['errors']:,}")
    print(f"[{year}] wrote                          : {year} 10k audit.json / {year} 10k audit.csv")
    print(f"[{year}] wrote isolated list            : {year} 10k isolated audit.json / {year} 10k isolated audit.csv")


def main():
    process_year(
        "2023",
        INPUT_DIRS["2023"],
        OUTPUT_DIRS["2023_cleaned"],
        OUTPUT_DIRS["2023_business"],
        OUTPUT_DIRS["2023_isolated"],
        limit=1000000,
    )
    process_year(
        "2024",
        INPUT_DIRS["2024"],
        OUTPUT_DIRS["2024_cleaned"],
        OUTPUT_DIRS["2024_business"],
        OUTPUT_DIRS["2024_isolated"],
        limit=1000000,
    )


if __name__ == "__main__":
    main()