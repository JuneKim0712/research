"""Compatibility wrapper; dropset now lives in org_mention_prefilter.py."""
from __future__ import annotations

from org_mention_prefilter import (
    DROPSPEC_EVIDENCE,
    DROPSPEC_VERSION,
    DROP_REGEX,
    EXACT_DROP_CASEFOLD,
    REVIEW_EXACT_CASEFOLD,
    WHITELIST_EXACT_CASEFOLD,
    export_dropset_snapshot,
)

__all__ = [
    "DROPSPEC_EVIDENCE",
    "DROPSPEC_VERSION",
    "DROP_REGEX",
    "EXACT_DROP_CASEFOLD",
    "REVIEW_EXACT_CASEFOLD",
    "WHITELIST_EXACT_CASEFOLD",
    "export_dropset_snapshot",
]


if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path(__file__).resolve().with_name("org_mention_junk_dropset.json")
    snap = export_dropset_snapshot()
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} (version {snap['version']})")
