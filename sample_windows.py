#!/usr/bin/env python3
"""Display 150 random samples from deduplicated windows."""

import json
import random
from pathlib import Path

jsonl_path = Path('combined_manifest_output_smoketest/abcd_random50/candidate_windows.jsonl')

with open(jsonl_path, encoding='utf-8') as f:
    windows = [json.loads(line) for line in f if line.strip()]

print(f"Total windows: {len(windows)}\n")

sample_size = min(150, len(windows))
sampled = random.sample(windows, sample_size)

print(f"Displaying {sample_size} random samples:")
print("=" * 130)

for i, w in enumerate(sampled, 1):
    company = w['source_company_name'][:38].ljust(38)
    cue_text = w['cue_text'][:32].ljust(32)
    hint = w['future_profile_hint'][:20].ljust(20)
    
    print(f"[{i:3d}] {w['window_id']} | {company} | T{w['cue_tier']} P{w['window_priority']} | {cue_text} | {hint}")
    
    window_preview = w['window_text'][:100].replace('\n', ' ')
    print(f"      {window_preview}...")
    print()

print("=" * 130)
print(f"Summary: Displayed {sample_size}/{len(windows)} deduplicated windows")
