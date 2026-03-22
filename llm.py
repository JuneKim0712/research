#!/usr/bin/env python3
"""
Gemini (free-tier friendly) LLM labeling pass for SEC mention rows **after** junk filtering.

For each CSV row, classifies the extracted mention using:
  mention_org + source_company + source_sentence

Writes the original columns plus:
    llm_label, llm_product_type, llm_owner_company_candidate, pipeline_role

Environment:
  GEMINI_API_KEY — required (Google AI Studio key)

Example:
  export GEMINI_API_KEY=...
  pip install google-generativeai
  python llm.py -i mentions.csv -o mentions_labeled.csv --test

One Gemini request per process (e.g. job array / serverless / strict rate caps):
  python llm.py -i mentions.csv -o mentions_labeled.csv \\
    --batch-size 1 --max-api-calls 1 --max-retries 1 --resume --no-repair
  # Re-run until every row has llm_label; --resume keeps prior labels. (--max-retries 1 avoids retry loops.)

Neat column order for sharing / Excel:
  python format_llm_mention_csv.py -i mentions_labeled.csv -o mentions_neat.csv --excel-bom

All rows in ONE Gemini request (one JSON ``items`` array), then map labels back to each CSV row:
  python llm.py -i mentions.csv -o mentions_labeled.csv --single-call --no-repair
  # Optional: --save-request-json request.json  OR  --save-bundle-csv bundle.csv (one cell: items_json)
  # ``row_index`` in the payload is local to that request (0..N-1 for uncached rows sent together).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Exact prompt text (also used as system instruction). Edit here only.
# ---------------------------------------------------------------------------

GEMINI_MENTION_LABEL_PROMPT = """You label ORG-NER spans from SEC 10-K competition-related windows.

For EACH item you receive, decide what the string `mention_org` refers to **in the context of** `source_sentence`,
with `source_company` as the filing company (the registrant). Additionally, when `mention_type` is supplied as "TICKER",
attempt to resolve the ticker symbol to a specific company name.

Output **only** valid JSON (no markdown fences, no commentary) with this exact top-level shape:
{"labels":[{"row_index":<int>,"llm_label":<string>,"llm_product_type":<string>,"llm_owner_company_candidate":<string>,"mention_type_resolved":<string>,"ticker_match_company":<string>},...]}

Rules for llm_label (use EXACTLY one of these strings):
- "COMPANY" — the span names a specific commercial enterprise (competitor, supplier, partner firm, named corporate entity).
- "PRODUCT_BRAND" — the span is primarily a branded product, drug name, device brand, or platform brand (not the legal company name by itself).
- "GENERIC_PRODUCT_CATEGORY" — generic sector/category/competitive bucket, not a specific firm (e.g. "large biopharmaceutical companies", "generic manufacturers").
- "Non coporate AGENCY" — government body, regulator, central bank, legislature/court/treaty-style actor, or other clearly non-commercial governmental / intergovernmental entity.
- "OTHER" — unclear, mixed, boilerplate fragment, non-entity noise, or anything you would not stake a downstream graph edge on. **Prefer OTHER when unsure.**

llm_owner_company_candidate:
- If llm_label is "PRODUCT_BRAND", set this to the best **specific** owning / marketing company name inferable from the sentence (or "" if none).
- For "COMPANY", repeat the same corporate name as in the span if it is the firm (or normalize lightly); use "" if not applicable.
- Otherwise usually "".

llm_product_type:
- If llm_label is "PRODUCT_BRAND" or "GENERIC_PRODUCT_CATEGORY", set this to a concise product category such as "drug", "vaccine", "medical device", "software platform", "service", or "consumer brand".
- If llm_label is "PRODUCT_BRAND" or "GENERIC_PRODUCT_CATEGORY" but type is unclear, set "unknown".
- For all non-product labels, set "".

mention_type_resolved:
- Echo back the NER-inferred mention_type if present and confident (COMPANY, PRODUCT, SERVICE, REGION, TICKER, INDUSTRY, TECHNOLOGY).
- If the NER type is TICKER but is not actually a ticker or cannot be matched, set to "NOT_TICKER".
- If no mention_type was provided, set to "".

ticker_match_company:
- If mention_type is "TICKER" and you can identify the company that the ticker represents, set this to the company name (or common name).
- Known tickers: AAPL=Apple, TSLA=Tesla, MSFT=Microsoft, GOOGL/GOOG=Alphabet, AMZN=Amazon, META=Meta, NVDA=Nvidia, AVGO=Broadcom, TXN=Texas Instruments, INTC=Intel, AMD=Advanced Micro Devices, AMAT=Applied Materials, LRCX=Lam Research, ASML=ASML, MU=Micron, SK Hynix, QCOM=Qualcomm, CDNS=Cadence, SNPS=Synopsys, ADBE=Adobe, CRM=Salesforce, NOW=ServiceNow, WDAY=Workday, ZM=Zoom, DOCU=DocuSign, NFLX=Netflix, DIS=Disney, PYPL=PayPal, SQ=Block, SHOP=Shopify, SPOT=Spotify, UBER=Uber, LYFT=Lyft, NOV=NOV (oil/gas), PTEN=Patterson-UTI, RRC=Range Resources, CVX=Chevron, XOM=ExxonMobil, COP=ConocoPhillips, MPC=Marathon Petroleum, PSX=Phillips 66, VLO=Valero, HES=Hess, EOG=EOG Resources, TRQ=Turquoise Hill, FCX=Freeport-McMoRan, MT=ArcelorMittal, US Steel=UnitedStates Steel, DD=DuPont, WM=Waste Management, ROK=Rockwell Automation, RTX=Raytheon, BA=Boeing, GE=General Electric, HON=Honeywell, JCI=Johnson Controls, EATON=Eaton, ABB=ABB, Siemens=SIE (German ADR), SAP=SAP, CSCO=Cisco, FFIV=F5, JNPR=Juniper, PALO=Palo Alto, OKTA=Okta, CRWD=CrowdStrike, PANW=Palo Alto Networks, FTNT=Fortinet, CHECK=Check Point, and many others.
- If no match or unclear, set to "" (empty).

Conservative: if the mention could be company or product, choose the label that fits the sentence best; if still ambiguous, use OTHER.

You MUST return one label object per input item, with matching row_index, in the SAME order as the input list."""

# ---------------------------------------------------------------------------
# Post-hoc pipeline mapping (deterministic from LLM fields)
# ---------------------------------------------------------------------------

VALID_LABELS = frozenset(
    {
        "COMPANY",
        "PRODUCT_BRAND",
        "GENERIC_PRODUCT_CATEGORY",
        "Non coporate AGENCY",
        "OTHER",
    }
)

LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "plc",
        "ag",
        "sa",
        "nv",
        "lp",
        "llp",
        "holdings",
        "holding",
        "group",
    }
)


def pipeline_role_from_llm(row: dict[str, Any]) -> str:
    """Map LLM fields to downstream routing bucket."""
    label = str(row.get("llm_label") or "").strip()
    owner = str(row.get("llm_owner_company_candidate") or "").strip()

    if label == "COMPANY":
        return "explicit_company_candidate"
    if label == "PRODUCT_BRAND" and owner:
        return "explicit_support_via_product"
    if label == "GENERIC_PRODUCT_CATEGORY":
        return "implicit_only"
    return "ignore_or_review"


def _norm_key(mention: str, company: str, sentence: str) -> str:
    payload = json.dumps(
        {"m": mention.strip(), "c": company.strip(), "s": sentence.strip()},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_cik(cik: str) -> str:
    digits = re.sub(r"\D", "", str(cik or ""))
    if not digits:
        return ""
    return str(int(digits))


def _name_keys(name: str) -> list[str]:
    base = (name or "").strip().lower()
    if not base:
        return []
    tokens = re.findall(r"[a-z0-9]+", base)
    if not tokens:
        return []

    keys: list[str] = []
    strict = " ".join(tokens)
    if strict:
        keys.append(strict)

    trimmed = list(tokens)
    while trimmed and trimmed[-1] in LEGAL_SUFFIXES:
        trimmed.pop()
    loose = " ".join(trimmed)
    if loose and loose != strict:
        keys.append(loose)
    return keys


def _build_company_cik_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_key: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cik = _normalize_cik(str(row.get("source_cik") or row.get("submitter_cik") or ""))
        if not cik:
            continue
        for name_col in ("source_company", "source_company_name"):
            nm = str(row.get(name_col) or "").strip()
            if not nm:
                continue
            for k in _name_keys(nm):
                by_key[k][cik] += 1

    resolved: dict[str, str] = {}
    for key, counts in by_key.items():
        if not counts:
            continue
        resolved[key] = counts.most_common(1)[0][0]
    return resolved


def _resolve_owner_cik(
    llm_blob: dict[str, Any],
    row: dict[str, Any],
    company_cik_lookup: dict[str, str],
) -> str:
    label = str(llm_blob.get("llm_label") or "").strip()
    if label not in ("COMPANY", "PRODUCT_BRAND"):
        return ""

    if label == "PRODUCT_BRAND":
        owner_name = str(llm_blob.get("llm_owner_company_candidate") or "").strip()
    else:
        owner_name = str(llm_blob.get("llm_owner_company_candidate") or "").strip() or str(
            row.get("mention_org") or ""
        ).strip()

    for key in _name_keys(owner_name):
        cik = company_cik_lookup.get(key)
        if cik:
            return cik
    return ""


def _load_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        return {}
    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_dir: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "cache_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)


def _cache_get(cache: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    blob = entry.get("blob")
    if isinstance(blob, dict):
        return blob
    return None


def _cache_put(cache: dict[str, dict[str, Any]], key: str, blob: dict[str, Any]) -> None:
    cache[key] = {"blob": blob}


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_label_response(raw: str, expected_indices: list[int]) -> dict[int, dict[str, Any]]:
    """Parse model JSON; validate indices and required keys."""
    cleaned = _strip_json_fence(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError("top-level object must contain 'labels' array")
    labels = data["labels"]
    if not isinstance(labels, list):
        raise ValueError("'labels' must be an array")

    by_idx: dict[int, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        ri = item.get("row_index")
        if not isinstance(ri, int):
            continue
        by_idx[ri] = item

    missing = [i for i in expected_indices if i not in by_idx]
    if missing:
        raise ValueError(f"missing row_index entries: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    out: dict[int, dict[str, Any]] = {}
    for i in expected_indices:
        obj = dict(by_idx[i])
        lab = str(obj.get("llm_label", "")).strip()
        if lab not in VALID_LABELS:
            raise ValueError(f"invalid llm_label for row_index={i}: {lab!r}")

        out[i] = {
            "llm_label": lab,
            "llm_product_type": str(obj.get("llm_product_type") or "").strip(),
            "llm_owner_company_candidate": str(obj.get("llm_owner_company_candidate") or "").strip(),
            "mention_type_resolved": str(obj.get("mention_type_resolved") or "").strip(),
            "ticker_match_company": str(obj.get("ticker_match_company") or "").strip(),
        }
    return out


def _items_from_batch_rows(batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Payload objects sent to Gemini (matches prompt schema)."""
    items: list[dict[str, Any]] = []
    for r in batch_rows:
        item_dict = {
            "row_index": r["_row_index"],
            "mention_org": r.get("mention_org", ""),
            "source_company": r.get("source_company", ""),
            "source_sentence": r.get("source_sentence", ""),
        }
        # Include mention_type if present (provides context to Gemini for classification)
        mention_type = str(r.get("mention_type", "")).strip()
        if mention_type:
            item_dict["mention_type"] = mention_type
        items.append(item_dict)
    return items


def _call_gemini_batch(
    genai: Any,
    model_name: str,
    batch_rows: list[dict[str, Any]],
    max_retries: int,
    temperature: float,
    *,
    allow_repair: bool = True,
) -> dict[int, dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=GEMINI_MENTION_LABEL_PROMPT)

    items = _items_from_batch_rows(batch_rows)
    indices = [int(r["_row_index"]) for r in batch_rows]

    user_payload = json.dumps({"items": items}, ensure_ascii=False)
    user_message = (
        "Classify each item in the following JSON. Return JSON only.\n\n" + user_payload
    )

    expected_indices = indices
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            raw = response.text or ""
            return _parse_label_response(raw, expected_indices)
        except Exception as e:
            last_err = str(e)
            if allow_repair:
                # Repair pass: ask again with the error (second HTTP request).
                repair = (
                    "Your previous output was invalid. Error:\n"
                    f"{last_err}\n\n"
                    "Return corrected JSON ONLY with the same schema, covering all row_index values: "
                    f"{expected_indices}.\n\nOriginal input JSON:\n{user_payload}"
                )
                try:
                    response2 = model.generate_content(
                        repair,
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=temperature * 0.5,
                        ),
                    )
                    raw2 = response2.text or ""
                    return _parse_label_response(raw2, expected_indices)
                except Exception as e2:
                    last_err = f"{e!s}; repair failed: {e2!s}"
            time.sleep(1.5 * (attempt + 1))
            continue

    raise RuntimeError(f"Gemini batch failed after {max_retries} attempts: {last_err}")


def _resolve_column(fieldnames: list[str], primary: str, fallbacks: list[str]) -> str:
    if primary in fieldnames:
        return primary
    for f in fallbacks:
        if f in fieldnames:
            return f
    raise SystemExit(
        f"Missing column {primary!r}. Tried fallbacks {fallbacks}. Found: {fieldnames[:40]}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Gemini LLM labeling for mention CSV rows.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Input CSV path")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output CSV path")
    p.add_argument(
        "--mention-col",
        default="mention_org",
        help="Column for extracted ORG span (default: mention_org; fallback: raw_mention)",
    )
    p.add_argument(
        "--source-company-col",
        default="source_company",
        help="Filing company column (default: source_company)",
    )
    p.add_argument(
        "--source-sentence-col",
        default="source_sentence",
        help="Sentence context column (default: source_sentence; fallback: sentence_text, trigger_sentence)",
    )
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model id (configurable)")
    p.add_argument("--batch-size", type=int, default=8, help="Rows per API call")
    p.add_argument(
        "--max-api-calls",
        type=int,
        default=0,
        help="Stop after this many Gemini requests (0 = no limit). Use 1 with --batch-size 1 for one request per run.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --output exists and has the same row count as --input, copy existing llm_* / pipeline_role into this run so you only label missing rows.",
    )
    p.add_argument("--test", action="store_true", help="Only process the first 20 data rows")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".gemini_mention_label_cache"),
        help="Disk cache directory for repeated (mention, company, sentence) tuples",
    )
    p.add_argument("--max-retries", type=int, default=4, help="Retries per batch on failure")
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="On bad JSON, do not send a second repair request (at most one generate_content per batch attempt).",
    )
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between batches (rate limits)",
    )
    p.add_argument(
        "--single-call",
        action="store_true",
        help="Send every uncached row in ONE Gemini request (ignores --batch-size). May hit context limits on large CSVs.",
    )
    p.add_argument(
        "--save-request-json",
        type=Path,
        default=None,
        help="Write the JSON body sent to Gemini (first API batch only; useful with --single-call).",
    )
    p.add_argument(
        "--save-bundle-csv",
        type=Path,
        default=None,
        help="Write a 1-row CSV with column items_json (same payload as --save-request-json) for Excel / inspection.",
    )
    args = p.parse_args()

    try:
        import google.generativeai as genai
    except ImportError:
        print("Install: pip install google-generativeai", file=sys.stderr)
        raise SystemExit(1)

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    with open(inp, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    mcol = _resolve_column(list(fieldnames), args.mention_col, ["raw_mention"])
    ccol = _resolve_column(list(fieldnames), args.source_company_col, ["source_company_name"])
    scol = _resolve_column(list(fieldnames), args.source_sentence_col, ["sentence_text", "trigger_sentence"])
    
    # Check if mention_type column exists (optional)
    mention_type_col = None
    for candidate in ["mention_type", "mention_types_union_filtered", "mention_types"]:
        if candidate in fieldnames:
            mention_type_col = candidate
            break

    if args.test:
        rows = rows[:20]

    cache_dir = args.cache_dir.expanduser().resolve()
    cache = _load_cache(cache_dir)

    extra_out = [
        "llm_label",
        "llm_product_type",
        "mention_type_resolved",
        "ticker_match_company",
        "llm_owner_company_candidate",
        "llm_owner_cik_candidate",
        "pipeline_role",
    ]
    out_fields = list(fieldnames) + [c for c in extra_out if c not in fieldnames]
    company_cik_lookup = _build_company_cik_lookup(rows)

    results_by_global: dict[int, dict[str, Any]] = {}
    if args.resume and out.is_file():
        with open(out, encoding="utf-8", newline="") as rf:
            prev_reader = csv.DictReader(rf)
            prev_rows = list(prev_reader)
        if len(prev_rows) == len(rows):
            for gi, prow in enumerate(prev_rows):
                lab = str(prow.get("llm_label") or "").strip()
                if not lab:
                    continue
                results_by_global[gi] = {
                    "llm_label": lab,
                    "llm_product_type": str(prow.get("llm_product_type") or "").strip(),
                    "llm_owner_company_candidate": str(
                        prow.get("llm_owner_company_candidate") or ""
                    ).strip(),
                    "llm_owner_cik_candidate": str(
                        prow.get("llm_owner_cik_candidate") or ""
                    ).strip(),
                    "mention_type_resolved": str(prow.get("mention_type_resolved") or "").strip(),
                    "ticker_match_company": str(prow.get("ticker_match_company") or "").strip(),
                }
        else:
            print(
                f"Warning: --resume ignored (output rows {len(prev_rows)} != input rows {len(rows)})",
                file=sys.stderr,
            )

    # Assign stable indices for batching
    work: list[tuple[int, dict[str, str], str]] = []
    for i, row in enumerate(rows):
        mention = str(row.get(mcol) or "").strip()
        company = str(row.get(ccol) or "").strip()
        sentence = str(row.get(scol) or "").strip()
        key = _norm_key(mention, company, sentence)
        work.append((i, row, key))

    n = len(work)
    # Process in batches (one chunk = up to batch_size rows, or entire table with --single-call)
    if args.single_call:
        batch_size = max(1, n)
        unc = 0
        for gidx, row, key in work:
            if gidx in results_by_global:
                continue
            if _cache_get(cache, key) is not None:
                continue
            unc += 1
        if unc > 200:
            print(
                f"Warning: --single-call will send {unc} rows in one request; "
                "very large inputs may exceed model context or time limits. "
                "Consider splitting the CSV or using default batching.",
                file=sys.stderr,
            )
        elif unc > 0:
            print(
                f"--single-call: one API request for {unc} uncached row(s) "
                f"({n} total rows; resume/cache may reduce this).",
                file=sys.stderr,
            )
    else:
        batch_size = max(1, int(args.batch_size))
    max_calls = int(args.max_api_calls)
    api_calls = 0

    i = 0
    pbar = tqdm(total=n, desc="LLM labeling") if tqdm is not None else None
    try:
        while i < n:
            chunk = work[i : i + batch_size]
            batch_meta: list[tuple[int, str]] = []
            batch_payload: list[dict[str, Any]] = []
            cache_hits: dict[int, dict[str, Any]] = {}

            for _off, (gidx, row, key) in enumerate(chunk):
                if gidx in results_by_global:
                    continue
                cached = _cache_get(cache, key)
                if cached is not None:
                    cache_hits[gidx] = cached
                    continue
                r = dict(row)
                r["_row_index"] = len(batch_payload)
                r["mention_org"] = str(row.get(mcol) or "").strip()
                r["source_company"] = str(row.get(ccol) or "").strip()
                r["source_sentence"] = str(row.get(scol) or "").strip()
                if mention_type_col:
                    r["mention_type"] = str(row.get(mention_type_col) or "").strip()
                batch_payload.append(r)
                batch_meta.append((gidx, key))

            for gidx, blob in cache_hits.items():
                results_by_global[gidx] = blob

            if batch_payload:
                if max_calls > 0 and api_calls >= max_calls:
                    break
                if api_calls == 0:
                    items_list = _items_from_batch_rows(batch_payload)
                    payload_obj = {"items": items_list}
                    if args.save_request_json:
                        pth = args.save_request_json.expanduser().resolve()
                        pth.parent.mkdir(parents=True, exist_ok=True)
                        pth.write_text(
                            json.dumps(payload_obj, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        print(f"Wrote request payload -> {pth}", file=sys.stderr)
                    if args.save_bundle_csv:
                        bpath = args.save_bundle_csv.expanduser().resolve()
                        bpath.parent.mkdir(parents=True, exist_ok=True)
                        with bpath.open("w", encoding="utf-8", newline="") as bf:
                            bw = csv.DictWriter(bf, fieldnames=["items_json"])
                            bw.writeheader()
                            bw.writerow(
                                {
                                    "items_json": json.dumps(
                                        payload_obj,
                                        ensure_ascii=False,
                                    )
                                }
                            )
                        print(f"Wrote 1-row bundle CSV -> {bpath}", file=sys.stderr)
                by_local = _call_gemini_batch(
                    genai,
                    args.model,
                    batch_payload,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    allow_repair=not args.no_repair,
                )
                api_calls += 1
                for j, (gidx, key) in enumerate(batch_meta):
                    blob = by_local[j]
                    _cache_put(cache, key, blob)
                    results_by_global[gidx] = blob
                _save_cache(cache_dir, cache)

            i += batch_size
            if pbar is not None:
                pbar.update(len(chunk))
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        if pbar is not None:
            pbar.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for gidx, row in enumerate(rows):
            out_row = dict(row)
            if gidx in results_by_global:
                llm = results_by_global[gidx]
                if not str(llm.get("llm_owner_cik_candidate") or "").strip():
                    llm["llm_owner_cik_candidate"] = _resolve_owner_cik(llm, row, company_cik_lookup)
                out_row.update(llm)
                out_row["pipeline_role"] = pipeline_role_from_llm(llm)
            else:
                out_row["llm_label"] = ""
                out_row["llm_product_type"] = ""
                out_row["mention_type_resolved"] = ""
                out_row["ticker_match_company"] = ""
                out_row["llm_owner_company_candidate"] = ""
                out_row["llm_owner_cik_candidate"] = ""
                out_row["pipeline_role"] = ""
            w.writerow(out_row)

    labeled = sum(1 for g in range(len(rows)) if g in results_by_global)
    msg = f"Wrote {len(rows)} rows -> {out} ({labeled} with llm_label)"
    if max_calls > 0 and labeled < len(rows):
        msg += f" [stopped after {api_calls} API call(s); rerun with --resume to continue]"
    print(msg)


if __name__ == "__main__":
    main()
