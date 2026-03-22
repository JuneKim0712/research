#!/usr/bin/env python3
"""
Generate a random 1000-window union sample from strict_explicit + contextual_explicit ABCD buckets.
Deduplicates on window_id, filters out ecosystem, saves to 2024/explicit_candidate_windows_1000.csv
"""

import pandas as pd
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
strict_path = Path("2024/2024_abcd_v1/candidate_windows_strict_explicit.csv")
contextual_path = Path("2024/2024_abcd_v1/candidate_windows_contextual_explicit.csv")
output_path = Path("2024/explicit_candidate_windows_1000.csv")

print(f"Loading strict_explicit from {strict_path}...")
df_strict = pd.read_csv(strict_path)
print(f"  Loaded {len(df_strict)} rows")

print(f"Loading contextual_explicit from {contextual_path}...")
df_contextual = pd.read_csv(contextual_path)
print(f"  Loaded {len(df_contextual)} rows")

# Union (keep all rows from both)
df_union = pd.concat([df_strict, df_contextual], ignore_index=True)
print(f"After union: {len(df_union)} rows")

# Deduplicate on window_id (keep first occurrence)
df_dedup = df_union.drop_duplicates(subset=["window_id"], keep="first")
print(f"After dedup on window_id: {len(df_dedup)} rows")

# Remove ecosystem cue phrases
df_no_ecosystem = df_dedup[df_dedup['cue_text'] != 'ecosystem'].copy()
print(f"After removing ecosystem: {len(df_no_ecosystem)} rows")

# Random sample of 1000
if len(df_no_ecosystem) < 1000:
    print(f"WARNING: Only {len(df_no_ecosystem)} unique windows, sampling all.")
    sample = df_no_ecosystem
else:
    sample = df_no_ecosystem.sample(n=1000, random_state=42)
    print(f"After random sample: {len(sample)} rows")

# Save
sample.to_csv(output_path, index=False)
print(f"\nSaved {len(sample)} windows to {output_path}")
