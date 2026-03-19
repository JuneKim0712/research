# SEC 10-K Filing Processing Pipeline

## Overview

Raw SEC 10-K filings are cleaned, parsed, and routed into output folders based on whether a usable business or competition section can be extracted.

---

## Output Folders

### `{year}_10K_cleaned/`
**Condition:** The filing was successfully processed and NOT isolated.  
Contains the cleaned plain-text version of every non-isolated filing.

---

### `{year}_10K_business/`
**Condition:** A business section (≥ 500 chars) or competition section (≥ 300 chars) was extracted, AND the filing was not sent to `no_outgoing_edges`.  
Contains `_business.txt` files with the extracted section text.

---

### `{year}_no_outgoing_edges/`
**Condition:** All of the following are true:
- A full business section was extracted (`section_source == "business"`, ≥ 500 chars)
- The filing was NOT isolated
- No competition section header was found (e.g. `Item 1 – Competition`)
- No competition terms (`competition`, `competitor`, `competitive`, `compete`, etc.) appear in the extracted business section

These filings mention no competitors in their business section, so they have no outgoing edges in a competition graph.

---

### `{year}_10K_isolated/`
Filings that could not produce a usable section are isolated here into one of the following subcategories:

#### `isolated/missing_both_no_competition_term/`
**Condition:** All of the following are true:
- No business section header detected
- No competition section header detected
- No competition terms found anywhere in the filing

The filing contains no recognizable section signals at all.

#### `isolated/no_business_header_has_competition_term/`
**Condition:** All of the following are true:
- No business section header detected
- No competition section header detected
- Competition terms are found somewhere in the filing

The filing mentions competition but has no extractable header anchor (neither business nor competition).

#### `isolated/business_header_only/`
**Condition:** Both of the following are true:
- A business or competition header was detected
- No extractable section content was found for those detected headers

At least one expected header exists, but the corresponding section content is missing or too short after cleaning.

---

## Routing Decision Order

```
Filing
│
├─ Check business header
│   └─ If business header exists, try business extraction (≥ 500 chars)
│
├─ If no business section extracted, check competition header
│   └─ If competition header exists, try competition extraction (≥ 300 chars)
│
├─ If a section was extracted (business OR competition)
│   ├─ If section_source == business AND no competition header AND no competition terms in extracted section
│   │   └─► {year}_no_outgoing_edges/  +  {year}_10K_cleaned/
│   └─ Otherwise
│       └─► {year}_10K_business/  +  {year}_10K_cleaned/
│
└─ If no section was extracted
    ├─ No business header + no competition header + competition terms anywhere
    │   └─► isolated/no_business_header_has_competition_term/
    ├─ No business header + no competition header + no competition terms anywhere
    │   └─► isolated/missing_both_no_competition_term/
    └─ Business header OR competition header exists, but no extractable content
        └─► isolated/business_header_only/
```

---

## Audit Files

| File | Description |
|---|---|
| `{year}_10k_audit.csv/.json` | Full audit log for every processed filing |
| `{year}_10k_isolated_audit.csv/.json` | Subset of audit log — isolated filings only |
| `header_only_business_filings_audited.csv/.json` | Detailed audit of `business_header_only` filings including recheck results |
| `2023_business_no_competition_term_list.csv/.json` | List of filings routed to `no_outgoing_edges` |
| `missing_business_audit_rechecked.csv/.json` | Recheck results for filings missing a business section |

---

## Section Extraction Thresholds

| Function | Min length | Max length |
|---|---|---|
| `extract_business_section()` | 500 chars | 500,000 chars |
| `extract_competition_section()` | 300 chars | 500,000 chars |
| `extract_business_header_only()` | 20 chars | 499 chars |

---

## Manifest Builder — `build_manifest.py`

### Purpose

`build_manifest.py` scans a folder of **already-extracted** filing text files
(e.g. `2024_10k_business/`) and produces a structured manifest in both **CSV**
and **JSON** format, plus a lightweight **parse-issues audit file**.

This script does **not** re-process raw filings, run NLP, split text into
windows, or infer any semantic content. It works only from the filenames and
raw character/line counts of the extracted text files.

---

### Usage

```bash
# Basic — process default inputs (2023 + 2024)
python build_manifest.py

# Single input folder
python build_manifest.py --input-dir 2024_10K_business

# Specify a separate output base directory
# (script creates year folders like 2024_manifest under this base)
python build_manifest.py --input-dir 2024_10K_business --output-dir ./manifests

# Also scan subdirectories (recursive)
python build_manifest.py --input-dir 2024_10K_business --recursive

# Change the large-file threshold (default: 200000 chars)
python build_manifest.py --input-dir 2024_10K_business --large-threshold 250000

# Optional random sampling
python build_manifest.py --input-dir 2023_10K_business --sample-size 200 --sample-seed 42
```

No third-party dependencies — only the Python standard library.

---

### Expected Filename Format

```
YYYY-MM-DD__COMPANY NAME__ACCESSION_sectiontype.txt
```

| Segment | Example | Notes |
|---|---|---|
| Date | `2024-01-31` | Leading `YYYY-MM-DD` |
| Company name | `COMCAST CORP` | Middle segment, between `__` delimiters |
| Accession | `0001166691-24-000011` | `##########-##-######` |
| Section type | `business` | Suffix after the final `_` and before `.txt` |

Full example: `2024-01-31__COMCAST CORP__0001166691-24-000011_business.txt`

The parser is robust to:
- Extra underscores or spaces around segment boundaries
- All-uppercase company names (auto title-cased for readability)
- Mixed-case company names (preserved as-is)
- Parenthetical suffixes like `(1)` — handled gracefully
- Missing section-type suffix — flagged in `parse_notes`

---

### Parsing Logic

| Field | Source | Fallback |
|---|---|---|
| `filing_date` | Leading `YYYY-MM-DD` in filename | — |
| `filing_year` | Derived from `filing_date` | — |
| `source_company_name` | Middle `__`-delimited segment | — |
| `accession_number` | Regex `\d{10}-\d{2}-\d{6}` in filename | — |
| `source_cik` | First 10-digit block of accession (integer form) | — |
| `cik_raw` | Same, with leading zeros preserved | — |
| `section_type` | Suffix after accession, before `.txt` | First 4 KB of file content |

The **content fallback** for `section_type` is triggered only if the filename
parse cannot find the field. It applies a simple regex scan over the first
4 KB of the file — no NLP, no windowing.

---

### Manifest Fields

#### Essential fields

| Field | Type | Description |
|---|---|---|
| `source_company_name` | str | Company name, lightly cleaned |
| `source_cik` | str | CIK as integer string (leading zeros stripped) |
| `filing_year` | str | 4-digit year derived from `filing_date` |
| `filing_date` | str | `YYYY-MM-DD` from filename |
| `accession_number` | str | Raw SEC accession, e.g. `0001166691-24-000011` |
| `original_filename` | str | Exact filename, unchanged |
| `section_type` | str | Extracted section label, e.g. `business` |

#### Helper fields

| Field | Type | Description |
|---|---|---|
| `file_path` | str | Absolute path to the file |
| `file_stem` | str | Filename without `.txt` extension |
| `text_char_count` | int | Total characters in the file |
| `text_line_count` | int | Total lines in the file |
| `is_large` | bool | `True` if char count > `--large-threshold` |
| `has_business_section` | bool | Content heuristic detected business section heading/text |
| `has_competition_section` | bool | Content heuristic detected competition section heading/text |
| `section_presence` | str | One of `both`, `business_only`, `competition_only`, `neither` |
| `parse_success` | bool | `True` if all essential fields were parsed confidently |
| `parse_notes` | str | Human-readable notes on any parse issues or fallbacks |
| `cik_raw` | str | CIK with leading zeros preserved (10 digits) |

---

### Output Files

| File | Description |
|---|---|
| `manifest.csv` | One row per `.txt` file, all fields above |
| `manifest.json` | Same data in JSON array format |
| `manifest_parse_issues.txt` | Lists every file where an essential field is missing or a fallback was used |

---

### Console Summary

After running, the script prints a short summary:

```
───────────────────────────────────────────────────────
  MANIFEST SUMMARY  —  /path/to/2024_10K_business
───────────────────────────────────────────────────────
  Total .txt files found     :  1,234
  Successfully parsed        :  1,228
  Failed / partial parses    :      6
  Large files                :      2  (> 200000 chars)
  Distinct section types     :      1
    • business                         1,228
───────────────────────────────────────────────────────
```

When scanning multiple input folders, output is written into year-specific
manifest folders, for example `2023_manifest/` and `2024_manifest/`.

---

## ABCD Candidate Windows + Audit Summary — `abcd.py`

### What this step does

`abcd.py` reads a manifest and the corresponding extracted text files, then
builds candidate text windows around competition-related cue phrases.

It writes three bucketed outputs:
- `candidate_windows_strict_explicit.*`
- `candidate_windows_contextual_explicit.*`
- `candidate_windows_broad_or_implicit.*`

and one audit report:
- `window_audit_summary.txt`

The script uses a tqdm progress bar during file scanning.

---

### How to read `window_audit_summary.txt`

#### File counts

- `Files in manifest (processed)`: total rows loaded from manifest.
- `Files with resolved text path`: rows that resolved to a valid input text path.
- `Files with at least one window`: files that produced one or more candidate windows.
- `Files with zero windows`: files that produced none.

#### Window counts

- `before dedup`: raw windows before overlap deduplication.
- `after dedup`: windows remaining after overlap-aware dedup.
- `Heading fallback windows`: windows created by heading fallback logic.
- `Files rescued by fallback`: files that had no cue windows but got at least one fallback window.

#### By cue_group

- `strict_explicit`: strongest direct competitor language.
- `contextual_explicit`: competitive context language, may be upgraded by heading/nearby strict cues.
- `implicit_or_broad`: broader/indirect competitive signals.
- `heading_fallback_broad`: fallback windows from competition-like headings when no cue windows exist.

#### By cue_tier (importance level)

- `Tier 4`: strict explicit cues.
- `Tier 3`: contextual explicit cues.
- `Tier 2`: implicit/broad cues.
- `Tier 1`: heading fallback windows.

#### Sentences harvested per trigger

ABCD uses different sentence window sizes depending on trigger type:

- `strict_explicit`:
  - default: 1 sentence (trigger only)
  - under competition heading: 3 sentences (previous + trigger + next)
- `contextual_explicit`:
  - default: 3 sentences (previous + trigger + next)
  - upgraded (under competition heading or near strict cue): up to 5 sentences
- `implicit_or_broad`:
  - 1 sentence (trigger only)
- `heading_fallback_broad`:
  - 1 sentence (single fallback sentence from heading block)

Global cap: at most 5 sentences per window.

#### By window_priority (ranking used in dedup preference)

- Higher is stronger.
- Typical mapping:
  - `5`: strict cue under competition heading (bonus)
  - `4`: strict cue
  - `3`: upgraded contextual cue
  - `2`: baseline contextual cue
  - `1`: implicit/broad cue
  - `0`: heading fallback cue

#### By section_type

Derived from content analysis per source file:
- `both`: business and competition sections detected.
- `business_only`: business detected, competition not detected.
- `competition_only`: competition detected, business not detected.
- `neither`: neither detected.

#### Top cue phrases

Most frequent matched cue text values among retained windows.

#### Top demotion reasons

Common contexts that suppress or downgrade weak/non-target cues.

---

### Fallback and related fields (definitions)

Fields appear in candidate window outputs:

- `is_heading_fallback`:
  - `true` if the window was created by heading fallback instead of a cue phrase.
- `fallback_reason`:
  - currently `competition_heading_without_explicit_cue` for fallback windows.
- `demotion_reason`:
  - reason a contextual signal was demoted/suppressed (for example generic or non-competitor context).

Fallback behavior:
- If a competition-like heading block has no regular cue windows, ABCD may create one very low-priority fallback window from that block.
- This preserves potentially useful competitive context without over-prioritizing it.

---

### Window output fields (quick definitions)

- `window_id`: stable window identifier.
- `cue_text`: exact matched phrase snippet.
- `cue_group`: strict/contextual/implicit/fallback grouping.
- `cue_tier`: 1-4 strength tier.
- `window_priority`: dedup ranking score.
- `trigger_sentence`: sentence that fired the cue/fallback.
- `window_text`: exported sentence window text.
- `local_heading_category`: nearest heading category.
- `export_bucket`: output bucket (`strict_explicit`, `contextual_explicit`, `broad_or_implicit`).
- `future_profile_hint`: heuristic profile hint for downstream enrichment.