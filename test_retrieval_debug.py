import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = r"data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
location = "aasld_full_site"
query = "when is the next conference date"

# Load models and index
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
index_file = os.path.join(BASE_DIR, location, "faiss_index.index")
metadata_file = os.path.join(BASE_DIR, location, "faiss_metadata.json")

print("Loading index...")
index = faiss.read_index(index_file)
with open(metadata_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ID mapping
def uuid_to_faiss_int(uid: str) -> int:
    return int(uid.replace('-', ''), 16) % (2**63 - 1)

int_id_to_uuid = {}
for uid in metadata:
    int_id = uuid_to_faiss_int(uid)
    int_id_to_uuid[int_id] = uid

# Search
print(f"Searching for: {query}")
query_embedding = model.encode([query]).astype("float32")
D, I = index.search(query_embedding, 10)

print("\nSearch Results:")
for i, int_id in enumerate(I[0]):
    if int_id == -1: continue
    uid = int_id_to_uuid.get(int_id)
    if not uid: continue
    item = metadata.get(uid)
    print(f"{i+1}. Score: {D[0][i]:.4f}")
    print(f"   Source: {item.get('sourceLink')}")
    print(f"   Text: {item.get('text')[:200]}...")
    print("-" * 20)
