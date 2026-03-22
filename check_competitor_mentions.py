import pandas as pd

df = pd.read_csv('2024/explicit_mentions_raw_2024_union1000_v2.csv')
competitor_rows = df[df['mention_types_raw'].str.contains('COMPETITOR_CATEGORY', na=False)]

print(f'Total COMPETITOR_CATEGORY mentions: {len(competitor_rows)}')
print('\nSample mentions (first 15):\n')

for idx, row in competitor_rows.head(15).iterrows():
    print(f"Source: {row['source_company']}")
    print(f"Mention: {row['org_mentions_raw']}")
    print()
