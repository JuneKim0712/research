#!/usr/bin/env python3
"""
Analyze deduplication logic to verify it's not inappropriately merging distinct windows.
Focuses on finding cases where one sentence names competitors and the next gives 
product-market context that should remain separate.
"""

import json
from pathlib import Path
from collections import defaultdict

# Character bigram overlap ratio (match the abcd.py logic)
def _overlap_ratio(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap on character bigrams to avoid tiny wording differences."""
    if not text_a or not text_b:
        return 0.0
    a = set(text_a[i : i + 2] for i in range(len(text_a) - 1))
    b = set(text_b[i : i + 2] for i in range(len(text_b) - 1))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load_windows(jsonl_path: Path) -> list[dict]:
    """Load all windows from JSONL file."""
    windows = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                windows.append(json.loads(line))
    return windows


def analyze_dedup_pattern():
    """Main analysis."""
    jsonl_path = Path('combined_manifest_output_smoketest/abcd_random50/candidate_windows.jsonl')
    windows = load_windows(jsonl_path)
    
    # Group windows by source file
    by_file = defaultdict(list)
    for w in windows:
        by_file[w['source_filename']].append(w)
    
    # Sort each file's windows by sentence index to see adjacency
    for fname in by_file:
        by_file[fname].sort(key=lambda x: x['start_sentence_idx'])
    
    print("=" * 100)
    print("DEDUPLICATION ANALYSIS: Checking for problematic merges")
    print("=" * 100)
    print()
    
    # Categories of potential problems
    adjacent_high_overlap = []
    competitor_mentions_separated = []
    
    for fname, file_windows in sorted(by_file.items()):
        # Look for consecutive windows (within 5 sentences)
        for i, w1 in enumerate(file_windows):
            for j, w2 in enumerate(file_windows[i+1:], i+1):
                idx_distance = w2['start_sentence_idx'] - w1['end_sentence_idx']
                
                # Only look at closely adjacent windows
                if idx_distance <= 5 and idx_distance >= 0:
                    overlap = _overlap_ratio(w1.get('window_text', ''), w2.get('window_text', ''))
                    
                    # Flag high overlap windows
                    if overlap >= 0.75:
                        cue1 = w1.get('cue_text', '')
                        cue2 = w2.get('cue_text', '')
                        hint1 = w1.get('future_profile_hint', '')
                        hint2 = w2.get('future_profile_hint', '')
                        
                        adjacent_high_overlap.append({
                            'file': fname,
                            'w1_id': w1['window_id'],
                            'w2_id': w2['window_id'],
                            'overlap': overlap,
                            'idx_distance': idx_distance,
                            'w1_cue': cue1,
                            'w2_cue': cue2,
                            'w1_hint': hint1,
                            'w2_hint': hint2,
                            'w1_text': w1.get('window_text', '')[:100],
                            'w2_text': w2.get('window_text', '')[:100],
                            'w1_trigger': w1.get('trigger_sentence', '')[:80],
                            'w2_trigger': w2.get('trigger_sentence', '')[:80],
                        })
                    
                    # Flag competitor mention followed by product-market distinction
                    if ('competitor' in cue1.lower() or 'compete' in cue1.lower()) and \
                       hint1 == 'competition_market' and hint2 != 'competition_market':
                        competitor_mentions_separated.append({
                            'file': fname,
                            'w1_id': w1['window_id'],
                            'w2_id': w2['window_id'],
                            'overlap': overlap,
                            'idx_distance': idx_distance,
                            'w1_cue': cue1,
                            'w2_cue': cue2,
                            'w1_hint': hint1,
                            'w2_hint': hint2,
                            'w1_trigger': w1.get('trigger_sentence', '')[:100],
                            'w2_trigger': w2.get('trigger_sentence', '')[:100],
                        })
    
    # Show adjacent high-overlap windows (potential merges)
    print(f"\n[1] ADJACENT WINDOWS WITH HIGH OVERLAP (>= 75%)")
    print(f"    These windows are in the same file and adjacent/close by sentence index")
    print(f"    If they have > 75% overlap, the dedup logic will keep only one.")
    print()
    print(f"    Found {len(adjacent_high_overlap)} pairs")
    print()
    
    if adjacent_high_overlap:
        for i, pair in enumerate(adjacent_high_overlap[:20], 1):
            print(f"    [{i}] File: {Path(pair['file']).name}")
            print(f"        Pair: {pair['w1_id']} <-> {pair['w2_id']} | Overlap: {pair['overlap']:.1%} | Distance: {pair['idx_distance']} sent")
            print(f"        W1 Cue: {pair['w1_cue']:<30} Hint: {pair['w1_hint']}")
            print(f"        W2 Cue: {pair['w2_cue']:<30} Hint: {pair['w2_hint']}")
            print(f"        W1 Trigger: {pair['w1_trigger']}")
            print(f"        W2 Trigger: {pair['w2_trigger']}")
            print()
        
        if len(adjacent_high_overlap) > 20:
            print(f"    ... and {len(adjacent_high_overlap) - 20} more")
    
    # Show competitor mentions followed by different context
    print()
    print(f"\n[2] COMPETITOR MENTION -> DIFFERENT CONTEXT PAIRS")
    print(f"    Look for: 'Sentence X names competitors' followed by 'Sentence Y gives product/geography/use-case context'")
    print(f"    These SHOULD be kept separate, not merged (if overlap is low).")
    print()
    print(f"    Found {len(competitor_mentions_separated)} pairs")
    print()
    
    if competitor_mentions_separated:
        for i, pair in enumerate(competitor_mentions_separated[:20], 1):
            print(f"    [{i}] File: {Path(pair['file']).name}")
            print(f"        Pair: {pair['w1_id']} <-> {pair['w2_id']} | Overlap: {pair['overlap']:.1%} | Distance: {pair['idx_distance']} sent")
            print(f"        W1 (competitor): {pair['w1_cue']:<30} Hint: {pair['w1_hint']}")
            print(f"        W2 (different):  {pair['w2_cue']:<30} Hint: {pair['w2_hint']}")
            print(f"        W1 Trigger: {pair['w1_trigger']}")
            print(f"        W2 Trigger: {pair['w2_trigger']}")
            print()
        
        if len(competitor_mentions_separated) > 20:
            print(f"    ... and {len(competitor_mentions_separated) - 20} more")
    
    # Overall statistics
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    
    total_before = sum(len(windows) for windows in by_file.values())
    total_deduplicated = sum(1 for w in windows if w.get('is_deduplicated'))
    print(f"Total windows in sample: {len(windows)}")
    print(f"Windows marked is_deduplicated=True: {total_deduplicated}")
    print(f"Dedup rate: {total_deduplicated / len(windows) * 100:.1f}%")
    print()
    
    # Distribution of overlap values for adjacent windows
    overlap_buckets = defaultdict(int)
    for pair in adjacent_high_overlap:
        bucket = f"{int(pair['overlap'] * 10) * 10}%"
        overlap_buckets[bucket] += 1
    
    print("Overlap distribution for adjacent high-overlap pairs:")
    for bucket in sorted(overlap_buckets.keys(), reverse=True):
        print(f"  {bucket}: {overlap_buckets[bucket]} pairs")
    print()


if __name__ == '__main__':
    analyze_dedup_pattern()
