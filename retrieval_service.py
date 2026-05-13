import os
# Bypass SSL verification for HF Hub in restricted environments
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
import sys
import json
import faiss
import numpy as np
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -----------------------------
# Config (must match ingestion_service.py)
# -----------------------------
BASE_DIR = os.getenv("FAISS_BASE_DIR", os.path.join(os.getcwd(), "data"))
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="FAISS Retrieval Service")

# -----------------------------
# Globals (Lazy loaded)
# -----------------------------
model = None
openai_client = None

def get_model():
    global model
    if model is None:
        print(f"Loading embedding model: {MODEL_NAME}...", file=sys.stderr)
        model = SentenceTransformer(MODEL_NAME)
    return model

def get_openai_client():
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
             print("Warning: OPENAI_API_KEY not set.", file=sys.stderr)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

def get_index_and_metadata_paths(location: str):
    location_dir = os.path.join(BASE_DIR, location.lower())
    index_file = os.path.join(location_dir, "faiss_index.index")
    metadata_file = os.path.join(location_dir, "faiss_metadata.json")
    return index_file, metadata_file

def safe_read_metadata(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading metadata {path}: {e}", file=sys.stderr)
        return {}

# -----------------------------
# Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    location: str = "aasld_full_site"
    top_k: int = 20  # Increased default for better coverage
    filters: Optional[Dict[str, Any]] = None  # e.g. {"audience": "member"}
    include_summary: bool = True

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_found: int
    summary: Optional[str] = None

# -----------------------------
# Logic
# -----------------------------
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    index_file, metadata_file = get_index_and_metadata_paths(request.location)
    
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        # Return empty if no index found for this location
        return {"results": [], "total_found": 0}

    # Load resources
    try:
        index = faiss.read_index(index_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")
        
    metadata = safe_read_metadata(metadata_file)
    
    # Embed query
    embedder = get_model()
    
    # Query Expansion for common intents
    expanded_query = request.query
    lower_q = request.query.lower()
    if any(kw in lower_q for kw in ["upcoming", "event", "webinar", "conference"]):
        # Add years and specific keywords to bridge semantic gap
        expanded_query += " 2025 2026 AASLD"
    
    query_embedding = embedder.encode([expanded_query]).astype("float32")
    
    # Search
    # Pull a much larger pool of candidates for internal re-ranking
    # since specific details can have high semantic distance from generic queries
    candidate_k = 500 
    D, I = index.search(query_embedding, candidate_k)
    
    # Build ID map
    def uuid_to_faiss_int(uid: str) -> int:
        return int(uid.replace('-', ''), 16) % (2**63 - 1)

    int_id_to_uuid = {}
    for uid, item in metadata.items():
        int_id = uuid_to_faiss_int(uid)
        int_id_to_uuid[int_id] = uid

    scored_candidates = []
    
    for i, int_id in enumerate(I[0]):
        if int_id == -1:
            continue
            
        uid = int_id_to_uuid.get(int_id)
        if not uid:
            continue
            
        item = metadata.get(uid)
        if not item:
            continue
            
        base_score = float(D[0][i])
        text_lower = item.get("text", "").lower()
        source = item.get("sourceLink", "")
        
        # KEYWORD BOOSTING / RE-RANKING
        # Note: Lower score is better in L2 FAISS
        boost = 0
        
        # Penalty for generic navigation chunks
        nav_keywords = ["skip to content", "main navigation", "family of websites", "navbar", "footer", "copyright 2026"]
        if any(nk in text_lower for nk in nav_keywords):
            boost -= 0.5 # Substantial penalty
        
        # Boost for future years
        if "2025" in text_lower: boost += 0.5
        if "2026" in text_lower: boost += 0.6
        
        # Boost for event-specific terms if they match query intent
        if any(kw in lower_q for kw in ["event", "webinar", "conference"]):
            if any(ew in text_lower for ew in ["register now", "emerging topic", "unified frontiers", "december 16", "resmetirom"]):
                boost += 0.8 # Significant boost for specific activities
            elif any(ew in text_lower for ew in ["webinar", "conference", "meeting"]):
                boost += 0.3
            
            if "events-and-webinars" in source:
                boost += 0.5 # Strong boost for the primary source
        
        final_score = base_score - boost
        
        # Filter
        if request.filters:
            match = True
            for k, v in request.filters.items():
                if item.get(k) != v:
                    match = False
                    break
            if not match:
                continue
        
        scored_candidates.append({
            "id": uid,
            "score": final_score,
            "text": item.get("text", ""),
            "metadata": {k:v for k,v in item.items() if k != "text"}
        })

    # Sort results by the new boosted score
    scored_candidates.sort(key=lambda x: x["score"])
    
    # Take more results for the LLM to process if it's a list query
    top_n = 25 if any(kw in lower_q for kw in ["list", "what are", "upcoming"]) else request.top_k
    final_results = [
        SearchResult(**res) for res in scored_candidates[:top_n]
    ]
            
    summary_text = None
    if request.include_summary:
        if not final_results:
             summary_text = "No results to summarize."
        else:
            try:
                client = get_openai_client()
                # Create context from results
                context_texts = [f"Result {i+1} (Source: {r.metadata.get('sourceLink', 'Unknown')}): {r.text}" for i, r in enumerate(final_results)]
                context = "\n\n".join(context_texts)
                
                print("DEBUG: Calling OpenAI for summarization...", file=sys.stderr)
                
                today_str = datetime.now().strftime("%B %d, %Y")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"""You are a helpful AASLD assistant. Today's date is {today_str}. 

CRITICAL INSTRUCTIONS FOR EVENTS:
1. When answering about conferences, webinars, or events, compare the event dates in the search results with today's date ({today_str}).
2. Identify whether an event is 'Upcoming' (occurs after today) or 'Past'.
3. PRIORITIZE SPECIFIC EVENT DETAILS. If one search result says "no events scheduled" (generic) but another specifies a title and a future date (e.g., in 2026), always TRUST THE SPECIFIC DATED EVENT.
4. The 'Events and Webinars' page is the primary source of truth for upcoming activities.
5. Provide a summary of the upcoming events you found, including their dates, titles, and links if available.
6. Use markdown hyperlinks for all links and email addresses.
7. If you find no specific future events at all after checking dates, only then state that none are scheduled."""},
                        {"role": "user", "content": f"Query: {request.query}\n\nSearch Results:\n{context}"}
                    ],
                    max_tokens=800,
                    temperature=0.3 # Lower temperature for better factual consistency
                )
                summary_text = response.choices[0].message.content
                print("DEBUG: OpenAI summarization successful.", file=sys.stderr)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"OpenAI Summarization failed: {e}", file=sys.stderr)
                summary_text = f"Error generating summary: {str(e)}"

    return {"results": final_results, "total_found": len(final_results), "summary": summary_text}

if __name__ == "__main__":
    import uvicorn
    # Use 8000 for retrieval as per chatbot_app.py config
    uvicorn.run(app, host="0.0.0.0", port=8000)
