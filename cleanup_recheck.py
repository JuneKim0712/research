import json
from pathlib import Path

# Read the no competition term list
json_file = Path('2023_business_no_competition_term_list.json')
with open(json_file, 'r', encoding='utf-8') as f:
    no_comp_list = json.load(f)

# Extract filenames that should be kept in the recheck folder
files_to_keep = set()
for entry in no_comp_list:
    business_file = entry.get('business_file', '')
    if '2023_10K_business_recheck' in business_file:
        # Extract just the filename
        filename = Path(business_file).name
        files_to_keep.add(filename)

print(f"Files to keep based on no competition list: {len(files_to_keep)}")

# List current files in the recheck folder
recheck_folder = Path('2023_10K_business_recheck')
current_files = set(f.name for f in recheck_folder.glob('*_business.txt'))
print(f"Current files in recheck folder: {len(current_files)}")

# Find files to delete
files_to_delete = current_files - files_to_keep
print(f"Files to delete: {len(files_to_delete)}")

# Delete files not in the list
for filename in files_to_delete:
    file_path = recheck_folder / filename
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted: {filename}")

print(f"\nCleanup complete!")
print(f"Remaining files in recheck folder: {len(list(recheck_folder.glob('*_business.txt')))}")
