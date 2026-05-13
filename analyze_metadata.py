import json
import os

metadata_path = r'c:\Users\ramya.s\Downloads\AASLD_scraping\data\aasld_full_site\faiss_metadata.json'

if os.path.exists(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    links = set()
    for item in metadata.values():
        if 'sourceLink' in item:
            links.add(item['sourceLink'])
    
    print(f"Total Unique URLs: {len(links)}")
    for link in sorted(list(links)):
        print(link)
else:
    print("Metadata file not found.")
