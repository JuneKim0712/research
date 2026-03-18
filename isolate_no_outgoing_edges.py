"""
Isolate files from 2023_10K_business that are listed in the no competition term list.
Move them to 2023_no_outgoing_edges folder so they ONLY exist there.
"""

import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "2023_business_no_competition_term_list.json"
BUSINESS_DIR = BASE_DIR / "2023_10K_business"
ISOLATED_DIR = BASE_DIR / "2023_no_outgoing_edges"

if not JSON_FILE.exists():
    raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")
if not BUSINESS_DIR.exists():
    raise FileNotFoundError(f"Business directory not found: {BUSINESS_DIR}")

ISOLATED_DIR.mkdir(parents=True, exist_ok=True)

# Load JSON to find files that should be in isolated folder
print(f"Loading JSON: {JSON_FILE}")
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    no_comp_list = json.load(f)

# Extract filenames that should be in 2023_no_outgoing_edges
files_to_move = set()
for entry in no_comp_list:
    business_file = entry.get('business_file', '')
    if '2023_no_outgoing_edges' in business_file or '2023_10K_business_recheck' in business_file:
        filename = Path(business_file).name
        files_to_move.add(filename)

print(f"Files to move to isolated folder: {len(files_to_move)}")

# Move files from 2023_10K_business to 2023_no_outgoing_edges
current_in_business = set(f.name for f in BUSINESS_DIR.glob('*_business.txt'))
files_in_both = files_to_move & current_in_business

print(f"Files currently in 2023_10K_business that should move: {len(files_in_both)}")

moved_count = 0
pbar = tqdm(files_in_both, desc="Moving files")
for filename in pbar:
    src_path = BUSINESS_DIR / filename
    dst_path = ISOLATED_DIR / filename
    
    if src_path.exists():
        # Move the file
        src_path.replace(dst_path)
        moved_count += 1
        pbar.set_postfix(moved=moved_count)

print(f"\nMove complete!")
print(f"  Files moved: {moved_count}")
print(f"  Remaining in 2023_10K_business: {len(current_in_business) - moved_count}")
print(f"  Total in 2023_no_outgoing_edges: {len(list(ISOLATED_DIR.glob('*_business.txt')))}")
