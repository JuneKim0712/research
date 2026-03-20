#!/usr/bin/env python3
"""
Gemini (free-tier friendly) LLM labeling pass for SEC mention rows **after** junk filtering.

For each CSV row, classifies the extracted mention using:
  mention_org + source_company + source_sentence

Writes the original columns plus:
  llm_label, llm_owner_company_candidate, llm_confidence, llm_reason, pipeline_role

Environment:
  GEMINI_API_KEY — required (Google AI Studio key)

Example:
  export GEMINI_API_KEY=...
  pip install google-generativeai
  python gemini_mention_label.py -i mentions.csv -o mentions_labeled.csv --test
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
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Exact prompt text (also used as system instruction). Edit here only.
# ---------------------------------------------------------------------------

GEMINI_MENTION_LABEL_PROMPT = """You label ORG-NER spans from SEC 10-K competition-related windows.

For EACH item you receive, decide what the string `mention_org` refers to **in the context of** `source_sentence`,
with `source_company` as the filing company (the registrant).

Output **only** valid JSON (no markdown fences, no commentary) with this exact top-level shape:
{"labels":[{"row_index":<int>,"llm_label":<string>,"llm_owner_company_candidate":<string>,"llm_confidence":<number>,"llm_reason":<string>},...]}

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

llm_confidence: a number from 0.0 to 1.0.

llm_reason: one short sentence, factual, citing cues from the sentence (no chain-of-thought).

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

        conf = obj.get("llm_confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise ValueError(f"llm_confidence must be a number for row_index={i}")
        c = float(conf)
        if c < 0.0 or c > 1.0:
            raise ValueError(f"llm_confidence out of range for row_index={i}")

        out[i] = {
            "llm_label": lab,
            "llm_owner_company_candidate": str(obj.get("llm_owner_company_candidate") or "").strip(),
            "llm_confidence": c,
            "llm_reason": str(obj.get("llm_reason") or "").strip(),
        }
    return out


def _call_gemini_batch(
    genai: Any,
    model_name: str,
    batch_rows: list[dict[str, Any]],
    max_retries: int,
    temperature: float,
) -> dict[int, dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=GEMINI_MENTION_LABEL_PROMPT)

    items = []
    indices: list[int] = []
    for r in batch_rows:
        indices.append(int(r["_row_index"]))
        items.append(
            {
                "row_index": r["_row_index"],
                "mention_org": r.get("mention_org", ""),
                "source_company": r.get("source_company", ""),
                "source_sentence": r.get("source_sentence", ""),
            }
        )

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
            # Repair pass: ask again with the error
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
    p.add_argument("--model", default="gemini-2.0-flash", help="Gemini model id (configurable)")
    p.add_argument("--batch-size", type=int, default=8, help="Rows per API call")
    p.add_argument("--test", action="store_true", help="Only process the first 20 data rows")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".gemini_mention_label_cache"),
        help="Disk cache directory for repeated (mention, company, sentence) tuples",
    )
    p.add_argument("--max-retries", type=int, default=4, help="Retries per batch on failure")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between batches (rate limits)",
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

    if args.test:
        rows = rows[:20]

    cache_dir = args.cache_dir.expanduser().resolve()
    cache = _load_cache(cache_dir)

    extra_out = [
        "llm_label",
        "llm_owner_company_candidate",
        "llm_confidence",
        "llm_reason",
        "pipeline_role",
    ]
    out_fields = list(fieldnames) + [c for c in extra_out if c not in fieldnames]

    # Assign stable indices for batching
    work: list[tuple[int, dict[str, str], str]] = []
    for i, row in enumerate(rows):
        mention = str(row.get(mcol) or "").strip()
        company = str(row.get(ccol) or "").strip()
        sentence = str(row.get(scol) or "").strip()
        key = _norm_key(mention, company, sentence)
        work.append((i, row, key))

    # Process in batches
    batch_size = max(1, int(args.batch_size))
    results_by_global: dict[int, dict[str, Any]] = {}

    i = 0
    n = len(work)
    while i < n:
        chunk = work[i : i + batch_size]
        batch_meta: list[tuple[int, str]] = []
        batch_payload: list[dict[str, Any]] = []
        cache_hits: dict[int, dict[str, Any]] = {}

        for _off, (gidx, row, key) in enumerate(chunk):
            cached = _cache_get(cache, key)
            if cached is not None:
                cache_hits[gidx] = cached
                continue
            r = dict(row)
            r["_row_index"] = len(batch_payload)
            r["mention_org"] = str(row.get(mcol) or "").strip()
            r["source_company"] = str(row.get(ccol) or "").strip()
            r["source_sentence"] = str(row.get(scol) or "").strip()
            batch_payload.append(r)
            batch_meta.append((gidx, key))

        for gidx, blob in cache_hits.items():
            results_by_global[gidx] = blob

        if batch_payload:
            by_local = _call_gemini_batch(
                genai,
                args.model,
                batch_payload,
                max_retries=args.max_retries,
                temperature=args.temperature,
            )
            for j, (gidx, key) in enumerate(batch_meta):
                blob = by_local[j]
                _cache_put(cache, key, blob)
                results_by_global[gidx] = blob
            _save_cache(cache_dir, cache)

        i += batch_size
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for gidx, row in enumerate(rows):
            out_row = dict(row)
            if gidx in results_by_global:
                llm = results_by_global[gidx]
                out_row.update(llm)
                out_row["pipeline_role"] = pipeline_role_from_llm(llm)
            else:
                out_row["llm_label"] = ""
                out_row["llm_owner_company_candidate"] = ""
                out_row["llm_confidence"] = ""
                out_row["llm_reason"] = ""
                out_row["pipeline_role"] = ""
            w.writerow(out_row)

    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
