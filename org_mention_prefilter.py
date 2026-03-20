"""
Label raw ORG mention strings before alias resolution / edge construction.

  keep — send downstream
  drop_obvious_junk — high-confidence trash from the in-file dropset
  review — ambiguous; keep out of automatic graph unless you promote via whitelist

Usage:
  from org_mention_prefilter import MentionFilter, label_mention, merge_dot_com_spans
  label_mention("Phase 1")  # ("drop_obvious_junk", "clinical_phase_prefix")

  # Per-window self-name suppression (pass filer name from CSV ``source_company``):
  MentionFilter().label("Acme Corp", source_company="Acme Corp, Inc.")

  f = MentionFilter(extra_whitelist={"my_org_alias"}, extra_drop_exact={"obvious typo inc"})
  f.label("my_org_alias")
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal

MentionLabel = Literal["keep", "drop_obvious_junk", "review"]

DROPSPEC_VERSION: Final[str] = "4.1.1"
DROPSPEC_EVIDENCE: Final[str] = (
    "raw.csv + nonraw.csv "
    "(RoBERTa NER, 2000 windows); inherits ORG_Strict_Raw_600* curation"
)

WHITELIST_EXACT_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        "x",
        "abi",
        "aws",
    }
)

EXACT_DROP_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        "company",
        "bank",
        "charter",
        "board",
        "committee",
        "registrant",
        "filing",
        "shareholder",
        "shareholders",
        "stockholder",
        "stockholders",
        "llc",
        "llp",
        "l.l.c.",
        "l.p.",
        "plc",
        "p.c.",
        "p.l.l.c.",
        "inc",
        "inc.",
        "corp",
        "corp.",
        "co.",
        "co",
        "ltd",
        "ltd.",
        "corporation",
        "incorporated",
        "limited",
        "nv",
        "n.v.",
        "s.a.",
        "ag",
        "kgaa",
        "sa",
        "internet",
        "european",
        "north american",
        "non-north american",
        "asian",
        "latin american",
        "middle eastern",
        "african",
        "scandinavian",
        "oceanian",
        "in",
        "chinese",
        "canadian",
        "spanish",
        "the internet",
        "fintech",
        "martech",
        "bdc",
        "hotel",
        "residential",
        "learners",
        "type",
        "lng",
        "eds",
        "operations",
        "functional",
        "global technology",
        "research and development",
        "human capital",
        "401k",
        "401(k)",
        "use authorization",
        "pneumatic comfort technologies",
        "acoustics",
        "south korean company",
        "commercial automobile program",
        "car's commercial automobile program",
        "the business combination",
        "business combination",
        "initial business combination",
        "initial public offering",
        "the initial public offering",
        "private placement",
        "private placement warrants",
        "public offering",
        "comprehensive plan for add",
        "supply chain services",
        "performance services",
        "com",
        "net",
        "org",
        "fda",
        "usda",
        "fha",
        "national cancer institute",
        "the national cancer institute",
        "federal reserve",
        "the federal reserve",
        "federal reserve bank",
        "the federal reserve bank",
        "u.s. international trade commission",
        "the u.s. international trade commission",
        "usitc",
        "european commission",
        "the european commission",
        "u.s. department of energy",
        "u.s. department of treasury",
        "the u.s. department of treasury",
        "u.s. navy",
        "u.s. government",
        "the u.s. government",
        "gse",
        "gses",
        "sec",
        "ftc",
        "irs",
        "epa",
        "osha",
        "nih",
        "cdc",
        "hhs",
        "cms",
        "ofac",
        "cftc",
        "finra",
        "occ",
        "fdic",
        "fcc",
        "fema",
        "dea",
        "atf",
        "fbi",
        "nhtsa",
        "uspto",
        "doj",
        "treasury",
        "the treasury",
        "medicare",
        "medicaid",
        "congress",
        "1940 act",
        "orange book",
        "the orange book",
        "covid-19",
        "lgbtq",
        "anda",
        "nda",
        "snda",
        "bla",
        "maa",
        "phase 1",
        "phase 2",
        "phase 3",
        "phase 1b",
        "phase 2b",
        "phase 3b",
        "cdk",
        "cdk9",
        "cd",
        "aml",
        "mll",
        "npm1",
        "pah",
        "ms",
        "btk",
        "pcnsl",
        "pcns",
        "bcl",
        "fl",
        "ce",
        "sars",
        "cov",
        "ec",
        "meni",
        "ccenture",
        "ffymetrix, inc",
        "ffymetrix, inc.",
        "ica biosystems, inc",
        "ica biosystems, inc.",
        "eophysical technologies",
        "uvasive, inc",
        "uvasive, inc.",
        "yndax pharmaceuticals, inc.",
        "yndax pharmaceuticals, inc",
        "ioenergy devco",
        "rbella mutual insurance company",
        "be",
        "cton dickinson",
        "rid dynamics",
        "knowb",
        "cy",
        "bersecurity",
        "sanita",
        "ware",
        "ava",
        "ility, llc",
        "fosys",
        "eistlich pharma ag",
        "ualtrics international inc",
        "ioventus inc",
        "rimient",
        "st",
        "rata oncology, inc",
        "pre",
        "therapeutics",
        "pt",
        "pa",
    }
)

DROP_REGEX: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"^\W+$", re.UNICODE), "punctuation_only"),
    (
        re.compile(
            r"^(?:[\*_]*)?"
            r"(?:"
            r"\(\s*[+\-\u2212]?\d[\d\s,\.\:/%_]*(?:[eE][+\-]?\d+)?\s*\)"
            r"|"
            r"[+\-\u2212]?\d[\d\s,\.\:/%_]*(?:[eE][+\-]?\d+)?"
            r")"
            r"[\*_]*$"
        ),
        "pure_numeric",
    ),
    (re.compile(r"^phase\s*\d+[a-z]?\b", re.I), "clinical_phase_prefix"),
    (re.compile(r"^phase\s+[ivx]+\b", re.I), "clinical_phase_roman"),
    (re.compile(r".*\bdose\s+escalation\b", re.I), "clinical_dose_escalation"),
    (re.compile(r"^early[\s-]stage\s+clinical\b", re.I), "clinical_early_stage"),
    (re.compile(r"\bclinical\s+trial\b", re.I), "clinical_trial_phrase"),
    (re.compile(r"^u\.s\.\s+department\s+of\s+", re.I), "us_department_of"),
    (re.compile(r"^the\s+u\.s\.\s+department\s+of\s+", re.I), "the_us_department_of"),
    (re.compile(r"(?i)\sact$"), "statute_act_suffix"),
    (re.compile(r"(?i)^stage\s*\d+[a-z]?\b"), "clinical_stage"),
    (re.compile(r"(?i)^cohort\s+[a-z0-9\-]+$"), "cohort_label"),
    (re.compile(r"(?i)^part\s+(\d+[a-z]?|[a-z]\d*)$"), "trial_part"),
    (re.compile(r"(?i)\bfda\s+approval\b"), "fda_approval_phrase"),
    (re.compile(r"(?i)^inhibitors?$"), "inhibitor_word"),
    (re.compile(r"(?i)^patients?$"), "patients_word"),
    (re.compile(r"(?i)\bgeneric\s+(?:drug|product|version)\b"), "generic_drug_phrase"),
    (re.compile(r"(?i)\bmarket\s+exclusivity\b"), "market_exclusivity_phrase"),
)

REVIEW_EXACT_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        "va",
        "car",
        "ai",
        "it",
        "hc",
        "sg",
        "lyr",
        "am",
        "dhcm",
        "dlf",
        "ge",
    }
)

_QUOTE_STRIP: Final[re.Pattern[str]] = re.compile(
    r'^[\s\"\'\u201c\u201d\u2018\u2019\u2022\u2023\u25e6\u2043\u2219]+|[\s\"\'\u201c\u201d\u2018\u2019\u2022\u2023\u25e6\u2043\u2219]+$'
)

# Strip list markers copied from slides/tables (e.g. ``• Application ; API ; …``).
_BULLET_CHARS: Final[str] = (
    "*"
    "\u2022\u2023\u2043\u2219\u25E6\u25AA\u25AB"  # bullets / small squares
    "\u00B7\u2024\u2218"  # middle dot, one-dot leader, ring operator
    "\u2013\u2014"  # en/em dash used as faux bullets
)
_BULLET_STRIP: Final[re.Pattern[str]] = re.compile(
    rf"^[{re.escape(_BULLET_CHARS)}\s]+|[{re.escape(_BULLET_CHARS)}\s]+$"
)

# Map curly/smart quotes and rare apostrophe code points to ASCII for stable matching.
_QUOTE_NORMALIZE: Final[dict[int, str]] = {
    0x2018: "'",
    0x2019: "'",
    0x201A: "'",
    0x201B: "'",
    0x2032: "'",
    0x02BC: "'",
    0x201C: '"',
    0x201D: '"',
    0x201E: '"',
    0x00AB: '"',
    0x00BB: '"',
}

# Strip from the *end* only; omit words that are real name tokens (e.g. "Trust" in "Northern Trust").
_LEGAL_TAIL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "inc",
        "inc.",
        "corp",
        "corp.",
        "corporation",
        "llc",
        "l.l.c.",
        "ltd",
        "ltd.",
        "lp",
        "llp",
        "plc",
        "co",
        "co.",
        "company",
        "limited",
        "sa",
        "s.a.",
        "ag",
        "nv",
        "n.v.",
    }
)

_TLD_MERGE: Final[frozenset[str]] = frozenset({"com", "net", "org", "io"})
_DOT_TLD: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?\.(?:com|net|org|io)$", re.I
)


def normalize_mention(text: str) -> str:
    """NFKC + normalize quotes/apostrophes to ASCII + trim + collapse whitespace; strip outer quotes and list bullets."""
    s = unicodedata.normalize("NFKC", text or "")
    s = "".join(_QUOTE_NORMALIZE.get(ord(c), c) for c in s)
    s = _QUOTE_STRIP.sub("", s)
    s = " ".join(s.split())
    s = _BULLET_STRIP.sub("", s).strip()
    s = " ".join(s.split())
    return s.strip()


def _casefold_key(norm: str) -> str:
    return norm.casefold()


def _company_fingerprint_for_self_match(text: str) -> str:
    """Strip legal tail tokens and punctuation noise; lowercase tokens for overlap checks."""
    s = normalize_mention(text)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = " ".join(s.split()).casefold()
    parts = s.split()
    while len(parts) > 1 and parts[-1] in _LEGAL_TAIL_TOKENS:
        parts.pop()
    return " ".join(parts)


def _contiguous_token_subspan(mention_fp: str, source_fp: str) -> bool:
    """True if mention tokens equal a contiguous run of source tokens."""
    m = mention_fp.split()
    s = source_fp.split()
    if not m or not s or len(m) > len(s):
        return False
    for i in range(len(s) - len(m) + 1):
        if s[i : i + len(m)] == m:
            return True
    return False


def _self_source_label(
    mention_norm: str,
    mention_key: str,
    source_company: str | None,
) -> tuple[MentionLabel, str] | None:
    """
    Drop or review when the mention is the filer name, a contiguous token fragment of it,
    or fingerprint-equal after stripping legal suffixes.
    """
    if not source_company or not str(source_company).strip():
        return None
    src_fp = _company_fingerprint_for_self_match(source_company)
    men_fp = _company_fingerprint_for_self_match(mention_norm)
    if not src_fp or not men_fp:
        return None

    if men_fp == src_fp:
        return "drop_obvious_junk", "self_source_full_match"

    if not _contiguous_token_subspan(men_fp, src_fp):
        return None

    m_toks = men_fp.split()
    s_toks = src_fp.split()
    if len(m_toks) >= len(s_toks):
        return None

    char_len = sum(len(t) for t in m_toks) + max(0, len(m_toks) - 1)
    if char_len >= 8:
        return "drop_obvious_junk", "self_source_fragment"
    return "review", "self_source_fragment_review"


def merge_dot_com_spans(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """
    Merge adjacent spans that are obvious broken ``*.com`` / ``*.net`` / … splits from NER.

    Expects (text, start, end) with end exclusive, sorted by start. Only merges when
    spans are adjacent (end == next start).
    """
    if len(spans) < 2:
        return list(spans)
    sp = sorted(spans, key=lambda x: (x[1], x[2]))
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(sp):
        t1, a1, b1 = sp[i]
        if i + 1 < len(sp):
            t2, a2, b2 = sp[i + 1]
            if b1 == a2:
                merged: str | None = None
                t1s, t2s = t1.strip(), t2.strip()
                if t1s.endswith(".") and t2s.casefold() in _TLD_MERGE:
                    merged = t1s + t2s
                elif t2s.startswith(".") and t2s.casefold() in {".com", ".net", ".org", ".io"}:
                    cand = t1s + t2s
                    if _DOT_TLD.match(cand.replace(" ", "")):
                        merged = cand
                elif "." in t1s and t2s.casefold() in _TLD_MERGE:
                    cand = t1s + t2s
                    if _DOT_TLD.match(cand.replace(" ", "")):
                        merged = cand
                if merged is not None and _DOT_TLD.match(merged.replace(" ", "")):
                    out.append((merged, a1, b2))
                    i += 2
                    continue
        out.append(sp[i])
        i += 1
    return out


@dataclass
class MentionFilter:
    """
    Stateful filter so you can extend the in-file dropset via runtime overrides.
    Later whitelists win over built-in drops.
    """

    extra_whitelist: set[str] = field(default_factory=set)
    extra_drop_exact: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._wl = {_casefold_key(normalize_mention(x)) for x in self.extra_whitelist}
        self._drop_extra = {_casefold_key(normalize_mention(x)) for x in self.extra_drop_exact}

    def label(
        self,
        raw_mention: str,
        *,
        source_company: str | None = None,
    ) -> tuple[MentionLabel, str]:
        norm = normalize_mention(raw_mention)
        if not norm:
            return "drop_obvious_junk", "empty_after_normalize"

        key = _casefold_key(norm)

        if key in self._wl or key in WHITELIST_EXACT_CASEFOLD:
            return "keep", "whitelist"

        if key in self._drop_extra:
            return "drop_obvious_junk", "extra_drop_exact"

        if key in EXACT_DROP_CASEFOLD:
            return "drop_obvious_junk", "exact_drop"

        # Regex drops before single-char heuristics (e.g. "." must match punctuation_only).
        for rx, slug in DROP_REGEX:
            if rx.search(norm):
                return "drop_obvious_junk", slug

        # Self-name: filer referring to itself (or a token chunk of its name) as ORG.
        self_lbl = _self_source_label(norm, key, source_company)
        if self_lbl is not None:
            return self_lbl

        if len(norm) == 1:
            if key == "x":
                return "keep", "single_letter_x_exception"
            if norm.isdigit():
                return "drop_obvious_junk", "single_digit"
            if norm.isalpha():
                return "drop_obvious_junk", "single_letter"
            return "review", "single_non_alnum"

        if key in REVIEW_EXACT_CASEFOLD:
            return "review", "review_exact"

        # Two-letter ALL CAPS Latin — often program/ticker noise; conservative review.
        if len(norm) == 2 and norm.isalpha() and norm.upper() == norm:
            return "review", "short_all_caps_2"

        return "keep", "default_keep"


_DEFAULT: Final[MentionFilter] = MentionFilter()


def label_mention(
    raw_mention: str,
    *,
    source_company: str | None = None,
) -> tuple[MentionLabel, str]:
    """Default global policy. Pass ``source_company`` from the window row to suppress self-filer mentions."""
    return _DEFAULT.label(raw_mention, source_company=source_company)


def dropspec_meta() -> dict[str, str]:
    return {
        "version": DROPSPEC_VERSION,
        "evidence": DROPSPEC_EVIDENCE,
    }


def export_dropset_snapshot() -> dict[str, object]:
    """Serializable snapshot for diffs / dashboards (regexes as source strings)."""
    return {
        "version": DROPSPEC_VERSION,
        "evidence": DROPSPEC_EVIDENCE,
        "whitelist_exact_casefold": sorted(WHITELIST_EXACT_CASEFOLD),
        "exact_drop_casefold": sorted(EXACT_DROP_CASEFOLD),
        "review_exact_casefold": sorted(REVIEW_EXACT_CASEFOLD),
        "drop_regex": [(p.pattern, slug) for p, slug in DROP_REGEX],
    }


if __name__ == "__main__":
    import argparse
    import csv
    from collections import Counter

    p = argparse.ArgumentParser(description="Audit labels on a window-level ORG CSV.")
    p.add_argument(
        "csv_path",
        nargs="?",
        default="ORG_Strict_Raw_600.csv",
        help="CSV with org_mentions_raw column",
    )
    p.add_argument(
        "--column",
        default="org_mentions_raw",
        help="Semicolon-separated mention column",
    )
    p.add_argument(
        "--source-column",
        default="",
        help="Optional column (e.g. source_company) for self-name suppression",
    )
    args = p.parse_args()
    sep = " ; "
    ctr: Counter[tuple[str, str]] = Counter()
    n_mentions = 0
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            col = (row.get(args.column) or "").strip()
            if not col:
                continue
            src = (row.get(args.source_column) or "").strip() if args.source_column else ""
            for part in col.split(sep):
                m = part.strip()
                if not m:
                    continue
                n_mentions += 1
                lbl, reason = (
                    label_mention(m, source_company=src) if src else label_mention(m)
                )
                ctr[(lbl, reason)] += 1
    print(dropspec_meta())
    print("total mention instances:", n_mentions)
    for (lbl, reason), c in ctr.most_common():
        print(f"{c:5d}  {lbl:18s}  {reason}")
