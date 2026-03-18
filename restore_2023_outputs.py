from __future__ import annotations

import csv
import html
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, FeatureNotFound
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SEC_DIR = BASE_DIR / "SEC"
INPUT_DIR = SEC_DIR / "2023_10K_raw"

OUTPUT_DIRS = {
    "cleaned": BASE_DIR / "2023_10K_cleaned",
    "business": BASE_DIR / "2023_10K_business",
    "isolated": BASE_DIR / "2023_10K_isolated",
    "no_outgoing_edges": BASE_DIR / "2023_no_outgoing_edges",
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


def read_text_robust(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_primary_10k_document(raw: str) -> str:
    docs = PRIMARY_DOC_RE.findall(raw)
    if not docs:
        return raw

    candidates = []
    for doc in docs:
        match = DOC_TYPE_RE.search(doc)
        doc_type = match.group(1).strip().upper() if match else ""
        if doc_type.startswith("10-K") or doc_type in {"10K", "10-K405", "10-KT", "10KSB"}:
            candidates.append(doc)

    chosen = max(candidates, key=len) if candidates else max(docs, key=len)
    text_match = TEXT_BLOCK_RE.search(chosen)
    return text_match.group(1) if text_match else chosen


def strip_non_text_blocks(raw: str) -> str:
    out = raw
    for pattern in (r"(?is)<PDF>.*?</PDF>", r"(?is)<EXCEL>.*?</EXCEL>", r"(?is)<ZIP>.*?</ZIP>"):
        out = re.sub(pattern, " ", out)
    return out


def strip_xbrl_support_blocks(raw: str) -> str:
    out = raw
    for pattern in XBRL_DROP_BLOCKS:
        out = re.sub(pattern, " ", out)
    return out


def is_hidden_tag(tag) -> bool:
    attrs = getattr(tag, "attrs", {}) or {}
    style = str(attrs.get("style", "")).replace(" ", "").lower()
    hidden_attr = str(attrs.get("hidden", "")).lower()
    aria_hidden = str(attrs.get("aria-hidden", "")).lower()
    return "display:none" in style or "visibility:hidden" in style or hidden_attr in {"hidden", "true"} or aria_hidden == "true"


def html_to_text(raw: str) -> str:
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
    stripped = line.strip()
    if not stripped:
        return False
    tokens = [token.strip(".,;:()[]{}") for token in stripped.split()]
    if len(tokens) < 6:
        return False
    junk = sum(1 for token in tokens if TAXONOMY_TOKEN_RE.fullmatch(token) or MEMBER_TOKEN_RE.fullmatch(token) or URL_TOKEN_RE.fullmatch(token))
    return (junk / max(len(tokens), 1)) >= 0.55


def is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in SEC_METADATA_LINE_PATTERNS + TOC_LINE_PATTERNS:
        if re.match(pattern, stripped, flags=re.I):
            return True
    return bool(re.match(r"^(page\s+)?\d{1,4}$", stripped, flags=re.I) or is_mostly_xbrl_noise(stripped))


def is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"(?i)^part\s+[ivx]+\b", stripped) or re.match(r"(?i)^item\s+\d+[a-z]?\b", stripped))


def rebuild_paragraphs(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line or is_junk_line(line):
            lines.append("" if not line else None)
        else:
            lines.append(line)

    paragraphs = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            paragraph = " ".join(buffer).strip()
            if paragraph:
                paragraphs.append(paragraph)
            buffer = []

    for line in lines:
        if line is None:
            continue
        if not line:
            flush()
        elif is_heading(line):
            flush()
            paragraphs.append(line.upper())
        else:
            buffer.append(line)
            if line.endswith((".", "!", "?", ":", ";")):
                flush()

    flush()
    return "\n\n".join(paragraphs).strip()


def trim_leading_preamble(text: str) -> str:
    starts = [match.start() for pattern in PREAMBLE_START_RES if (match := pattern.search(text)) and match.start() > 1000]
    return text[min(starts):].strip() if starts else text


def clean_filing(raw_text: str) -> str:
    raw_text = extract_primary_10k_document(raw_text)
    text = html_to_text(raw_text)
    text = normalize_unicode_and_whitespace(text)
    text = rebuild_paragraphs(text)
    return trim_leading_preamble(text)


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
    for _, start_re in start_patterns:
        candidates = []
        for match in start_re.finditer(text):
            start = match.start()
            surrounding = text[max(0, start - 5): match.end() + 20]
            if re.search(r"\.{3,}|\t\d{1,3}\s*$", surrounding, flags=re.I):
                continue

            ends = []
            for end_re in end_patterns:
                end_match = end_re.search(text, pos=match.end() + 1)
                if end_match and end_match.start() > start:
                    ends.append(end_match.start())

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


def iter_input_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def backup_existing_outputs() -> Optional[Path]:
    existing = [path for path in OUTPUT_DIRS.values() if path.exists()]
    if not existing:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = BASE_DIR / "recovery_backups" / f"2023_restore_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    for path in existing:
        shutil.move(str(path), str(backup_root / path.name))

    return backup_root


def process_file(src_path: Path) -> dict:
    raw = read_text_robust(src_path)
    cleaned = clean_filing(raw)
    has_business_header = any(start_re.search(cleaned) for _, start_re in BUSINESS_START_PATTERNS)
    has_competition_header = any(start_re.search(cleaned) for _, start_re in COMPETITION_START_PATTERNS)
    has_competition_term_anywhere = bool(COMPETITION_TERM_RE.search(cleaned))

    section_text: Optional[str] = None
    section_source: Optional[str] = None
    audit_category: Optional[str] = None
    audit_reason: Optional[str] = None
    isolate = False

    if not has_business_header and not has_competition_header and not has_competition_term_anywhere:
        isolate = True
        audit_category = "missing_both_no_competition_term"
        audit_reason = "No business/competition section and no competition term found."
    elif has_competition_header and not has_business_header and not has_competition_term_anywhere:
        isolate = True
        audit_category = "competition_header_only_no_term"
        audit_reason = "Competition section header found but no business header and no competition terms."
    elif not has_business_header and has_competition_term_anywhere:
        isolate = True
        audit_category = "no_business_header_has_competition_term"
        audit_reason = "No business header detected but competition terms found elsewhere."
    else:
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
                audit_reason = "Business header found but content too short (likely over-cleaned)."

    has_competition_term_in_business = bool(COMPETITION_TERM_RE.search(section_text)) if section_source == "business" and section_text else False

    cleaned_name = src_path.stem + ".txt"
    section_name = src_path.stem + "_business.txt"
    cleaned_path = OUTPUT_DIRS["cleaned"] / cleaned_name
    business_path = OUTPUT_DIRS["business"] / section_name
    no_outgoing_path = OUTPUT_DIRS["no_outgoing_edges"] / section_name

    if isolate:
        cat_dir = OUTPUT_DIRS["isolated"] / str(audit_category)
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / cleaned_name).write_text(cleaned, encoding="utf-8")
        if section_text:
            (cat_dir / section_name).write_text(section_text, encoding="utf-8")
    elif section_text and section_source == "business" and not has_competition_header and not has_competition_term_in_business:
        cleaned_path.write_text(cleaned, encoding="utf-8")
        no_outgoing_path.write_text(section_text, encoding="utf-8")
    else:
        cleaned_path.write_text(cleaned, encoding="utf-8")
        if section_text:
            business_path.write_text(section_text, encoding="utf-8")

    return {
        "source_file": str(src_path),
        "cleaned_file": str(cleaned_path) if not isolate else None,
        "business_file": str(business_path) if (section_text and not isolate and not (section_source == "business" and not has_competition_header and not has_competition_term_in_business)) else None,
        "no_outgoing_edges_file": str(no_outgoing_path) if (section_text and section_source == "business" and not isolate and not has_competition_header and not has_competition_term_in_business) else None,
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


def write_audit_files(manifest: list[dict]) -> None:
    out_json = BASE_DIR / "2023_10k_audit.json"
    out_csv = BASE_DIR / "2023_10k_audit.csv"

    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred_fields = [
        "source_file", "cleaned_file", "business_file", "no_outgoing_edges_file",
        "cleaned_chars", "has_business_section", "business_chars",
        "has_competition_term_anywhere", "has_competition_term_in_business",
        "section_source", "audit_category", "audit_reason", "is_isolated",
    ]
    seen = set(preferred_fields)
    extra_fields = [key for row in manifest for key in row.keys() if key not in seen and (seen.add(key) or True)]

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred_fields + extra_fields)
        writer.writeheader()
        writer.writerows(manifest)


def write_isolated_audit(manifest: list[dict]) -> None:
    isolated = [row for row in manifest if row.get("is_isolated")]
    out_json = BASE_DIR / "2023_10k_isolated_audit.json"
    out_csv = BASE_DIR / "2023_10k_isolated_audit.csv"

    out_json.write_text(json.dumps(isolated, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "audit_category", "audit_reason", "source_file", "cleaned_chars",
        "has_business_section", "has_competition_term_anywhere", "section_source", "is_isolated",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(isolated)


def write_category_lists(manifest: list[dict]) -> None:
    categories: dict[str, list[dict]] = {}
    for row in manifest:
        if row.get("is_isolated"):
            category = row.get("audit_category", "unknown")
            categories.setdefault(str(category), []).append(row)

    for category, rows in categories.items():
        out_file = BASE_DIR / f"2023_{category}_list.json"
        out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Missing input folder: {INPUT_DIR}")

    backup_root = backup_existing_outputs()
    for directory in OUTPUT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(INPUT_DIR))
    manifest: list[dict] = []
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

    for src_path in tqdm(files, desc="[2023] Restoring", unit="file"):
        try:
            meta = process_file(src_path)
            manifest.append(meta)

            if meta.get("no_outgoing_edges_file"):
                counts["no_outgoing_edges"] += 1
            elif meta.get("section_source") == "business":
                counts["business"] += 1
            elif meta.get("section_source") == "competition":
                counts["competition"] += 1

            if meta.get("is_isolated"):
                counts[str(meta.get("audit_category", "unknown"))] = counts.get(str(meta.get("audit_category", "unknown")), 0) + 1
        except Exception as exc:
            manifest.append({"source_file": str(src_path), "error": str(exc)})
            counts["errors"] += 1

    write_audit_files(manifest)
    write_isolated_audit(manifest)
    write_category_lists(manifest)

    summary = {
        "backup_root": str(backup_root) if backup_root else None,
        "total_files": len(files),
        "business": counts["business"],
        "competition": counts["competition"],
        "no_outgoing_edges": counts["no_outgoing_edges"],
        "missing_both_no_competition_term": counts["missing_both_no_competition_term"],
        "competition_header_only_no_term": counts["competition_header_only_no_term"],
        "no_business_header_has_competition_term": counts["no_business_header_has_competition_term"],
        "business_header_only": counts["business_header_only"],
        "errors": counts["errors"],
    }
    (BASE_DIR / "2023_restore_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()