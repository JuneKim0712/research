import json
from pathlib import Path

JSON_FILE = Path("2023_business_no_competition_term_list.json")

# Load and update
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Replace old folder path with new one
updated = 0
for entry in data:
    if '2023_10K_business_recheck' in entry.get('business_file', ''):
        entry['business_file'] = entry['business_file'].replace('2023_10K_business_recheck', '2023_no_outgoing_edges')
        updated += 1

# Save updated JSON
with open(JSON_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Updated {updated} entries in JSON file")
