"""
Conservative ORG junk dropset for raw mentions (pre–alias resolution, pre–edges).

Policy (enforced together with ``org_mention_prefilter``):

- **Single-character mentions** — drop all except **X** (whitelisted; e.g. Twitter/X).
- **Standalone punctuation / symbols** — drop (no letters or digits).
- **Pure numbers** — drop (including numeric strings with grouping punctuation).
- **Apostrophes / quotes** — normalized to ASCII in ``normalize_mention`` for stable matching.
- **Broken ``.com``** — standalone ``com`` / ``net`` / ``org`` dropped; use ``merge_dot_com_spans`` on offset spans to rejoin ``Bill.``+``com``.
- **Generic filing / governance words** — e.g. company, bank, charter, board (not firm names).
- **Biotech / regulatory phrases** — inhibitors, patients, FDA approval, ANDA/NDA/BLA, etc.
- **SPAC / capital-markets boilerplate** — IPO, business combination, private placement warrants, …
- **Service / category phrases** — e.g. supply chain services, performance services.
- **Regional / sector descriptors** — European, North American, Asian, …
- **Government agencies & statute-style names** — agencies by exact token; ``… act`` by regex.
- **Clinical stage / trial-part / cohort-style labels** — regex; whitelist exceptions.
- **Self-filer suppression** — optional ``source_company`` on ``MentionFilter.label`` / ``label_mention`` drops or reviews mentions that are the filer name or a contiguous token fragment.

Grounded in RoBERTa 2000-window exports plus earlier 600-window strict runs.

This module is data-only (no I/O). Edit lists here or extend via
``org_mention_prefilter.MentionFilter(extra_whitelist=..., extra_drop_exact=...)``.

Version bump when you change rules in a material way.

Standalone numbers use a dedicated regex (signs, grouping, times, ratios, ``%``, scientific ``e``, light ``*``/``_`` wrappers); see ``pure_numeric`` in ``DROP_REGEX``.
"""
from __future__ import annotations

import re
from typing import Final

DROPSPEC_VERSION: Final[str] = "4.1.1"
DROPSPEC_EVIDENCE: Final[str] = (
    "org_detection_mentions_2000_raw.csv + org_detection_mentions_2000_nonraw.csv "
    "(RoBERTa NER, 2000 windows); inherits ORG_Strict_Raw_600* curation"
)

# --- Whitelist (case-insensitive match on normalized text) -----------------
# Keep X for Twitter/X; keep a few high-signal acronyms that are real competitors
# in this corpus and are easily confused with junk rules.
WHITELIST_EXACT_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        "x",
        "abi",  # ABI / Anheuser-Busch InBev shorthand in beer competitive set
        "aws",  # Amazon Web Services
    }
)

# --- Exact drops (case-insensitive); whole mention must match after normalize -----
# Built from recurring tokens in the 600-sample raw ORG column (frequency + manual curation).
EXACT_DROP_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        # Generic legal / entity shells & governance boilerplate (non-specific firm)
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
        "nv",  # legal-form fragment from "SA/NV" splits (Molson Coors window)
        "n.v.",
        "s.a.",
        "ag",
        "kgaa",
        "sa",
        # Non-company descriptors mis-tagged as ORG in the 600 run
        "internet",
        "european",
        "north american",
        "asian",
        "latin american",
        "middle eastern",
        "african",
        "scandinavian",
        "oceanian",
        "in",  # sentence adverb / noise ORG
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
        # SPAC / filing boilerplate & truncated gov-doc titles (RoBERTa 2000 sample)
        "the business combination",
        "business combination",
        "initial business combination",
        "initial public offering",
        "the initial public offering",
        "private placement",
        "private placement warrants",
        "public offering",
        "comprehensive plan for add",
        # Service / category phrases (not competitor firm names as single ORG spans)
        "supply chain services",
        "performance services",
        # Stray TLD fragments from broken ``*.com`` / ``*.net`` splits (see merge_dot_com_spans)
        "com",
        "net",
        "org",
        # Government / public bodies (whitelist later if you want a node)
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
        # Federal / common agencies (token is whole mention; not resolved acronyms in longer names)
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
        # Public payers, legislature, laws & publications (not corporate competitors)
        "medicare",
        "medicaid",
        "congress",
        "1940 act",
        "orange book",
        "the orange book",
        "covid-19",
        "lgbtq",
        # Abbreviated regulatory / application types (whole-span ORG noise in R&D windows)
        "anda",
        "nda",
        "snda",
        "bla",
        "maa",
        # Clinical / biology / program noise common in the pharma windows
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
        "ms",  # disease acronym in PCNSL/MS windows (not Microsoft)
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
        # NER split / corruption fragments seen in org_diff + raw aggregates
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
        "st",  # stray headline fragment paired with "Strata Oncology" splits
        "rata oncology, inc",
        "pre",  # prelude therapeutics split
        "therapeutics",  # lone suffix after "Pre" split — too generic alone
        "pt",  # PTEFb fragment
        "pa",  # stray fragment in Aimmune window
    }
)

# --- Regex drops: (compiled_pattern, reason_slug) ----------------------------
# Whole-string match on normalized mention unless pattern uses ^...$ internally.
DROP_REGEX: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Lone punctuation / symbols (e.g. "." ×460 in RoBERTa 2000 raw); not words/digits.
    (re.compile(r"^\W+$", re.UNICODE), "punctuation_only"),
    # Standalone numbers: sign; ``(…)``; ``, . : / % _`` grouping; optional ``1e6``; stray ``*``/``_`` wrappers.
    # Does not match tokens with letters (e.g. ``3M``) except scientific ``e`` / ``E``.
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
    # Statute-style titles ending in " … Act" (avoids single words like "abstract" / "compact").
    (re.compile(r"(?i)\sact$"), "statute_act_suffix"),
    # Trial design / stage labels (not company names); whitelist exceptions.
    (re.compile(r"(?i)^stage\s*\d+[a-z]?\b"), "clinical_stage"),
    (re.compile(r"(?i)^cohort\s+[a-z0-9\-]+$"), "cohort_label"),
    (re.compile(r"(?i)^part\s+(\d+[a-z]?|[a-z]\d*)$"), "trial_part"),
    # Biotech / regulatory wording (not competitor names)
    (re.compile(r"(?i)\bfda\s+approval\b"), "fda_approval_phrase"),
    (re.compile(r"(?i)^inhibitors?$"), "inhibitor_word"),
    (re.compile(r"(?i)^patients?$"), "patients_word"),
    (re.compile(r"(?i)\bgeneric\s+(?:drug|product|version)\b"), "generic_drug_phrase"),
    (re.compile(r"(?i)\bmarket\s+exclusivity\b"), "market_exclusivity_phrase"),
)

# --- Review heuristics: short ambiguous tokens (not auto-dropped) ----------
# If mention casefold matches, force review (unless whitelisted).
REVIEW_EXACT_CASEFOLD: Final[frozenset[str]] = frozenset(
    {
        "va",  # VA loan program vs Virginia
        "car",  # residual market acronym vs company
        "ai",
        "it",
        "hc",
        "sg",
        "lyr",
        "am",
        "dhcm",
        "dlf",
        "ge",  # could be General Electric fragment vs German "AG" noise
    }
)

# 2–3 character all-caps-ish tokens default to review (conservative).
REVIEW_SHORT_MAX_LEN: Final[int] = 3


if __name__ == "__main__":
    import json
    from pathlib import Path

    from org_mention_prefilter import export_dropset_snapshot

    out = Path(__file__).resolve().with_name("org_mention_junk_dropset.json")
    snap = export_dropset_snapshot()
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} (version {snap['version']})")
