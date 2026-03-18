import json
from pathlib import Path

# Update JSON file references
json_file = Path('2023_business_no_competition_term_list.json')
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

updated_count = 0
for entry in data:
    if 'business_file' in entry and 'no_outgoing_edges' in entry['business_file']:
        entry['business_file'] = entry['business_file'].replace('no_outgoing_edges', '2023_no_outgoing_edges')
        updated_count += 1

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated {updated_count} entries in 2023_business_no_competition_term_list.json")
