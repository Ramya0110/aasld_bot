# app/main.py
import os
import io
import sys
import uuid
import json
import re
import time
import fitz  # PyMuPDF
import numpy as np
import faiss
from PIL import Image
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available, will use text extraction only", file=sys.stderr)
from typing import List, Tuple, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from scraper_utils import WebScraper
import base64
import pandas as pd
import mimetypes
 
from sentence_transformers import SentenceTransformer
import nltk
 
# -----------------------------
# Config (env-overridable)
# -----------------------------
BASE_DIR = os.getenv("FAISS_BASE_DIR", os.path.join(os.getcwd(), "data"))
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")  # Upgraded for better semantic understanding
DEFAULT_DPI = int(os.getenv("OCR_DPI", "100"))  # For image rendering when needed
OCR_LANG = os.getenv("OCR_LANG", "en")  # EasyOCR uses 'en' not 'eng'
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_MAX_PAGES = int(os.getenv("DEFAULT_MAX_PAGES", "0")) # 0 means unlimited
os.makedirs(BASE_DIR, exist_ok=True)
 
# Initialize EasyOCR reader if available (lazy loading)
ocr_reader = None
if EASYOCR_AVAILABLE:
    try:
        ocr_reader = easyocr.Reader([OCR_LANG], gpu=False)  # CPU mode for compatibility
        print(f"EasyOCR initialized with language: {OCR_LANG}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize EasyOCR: {e}", file=sys.stderr)
        EASYOCR_AVAILABLE = False
 
 
# Load the embedding model once at startup
model = SentenceTransformer(MODEL_NAME)
 
# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...", file=sys.stderr)
    nltk.download('punkt', quiet=True)
 
app = FastAPI(title="PDF OCR + FAISS Ingestion API (with JSON support)")
 
# Add CORS middleware to allow n8n to connect
from fastapi.middleware.cors import CORSMiddleware
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
 
# -----------------------------
# Helpers
# -----------------------------
def get_index_and_metadata_paths(location: str) -> Tuple[str, str]:
    """Return index & metadata path for location."""
    # Sanitize location for Windows paths
    safe_location = re.sub(r'[<>:"/\\|?*]', '_', location.lower())
    location_dir = os.path.join(BASE_DIR, safe_location)
    os.makedirs(location_dir, exist_ok=True)
    index_file = os.path.join(location_dir, "faiss_index.index")
    metadata_file = os.path.join(location_dir, "faiss_metadata.json")
    return index_file, metadata_file
 
def get_index_and_metadata_paths_for_json(location: str) -> Tuple[str, str]:
    """Use separate directory suffix for JSON-ingested content."""
    # e.g., location 'apac' -> 'apac_json'
    json_loc = f"{location.lower()}"
    return get_index_and_metadata_paths(json_loc)
 
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Legacy fixed-size chunking (kept for backward compatibility)."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    # sentence-aware chunking (simple)
    parts = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        parts.append(clean_text[start:end].strip())
        start += max(1, chunk_size - overlap)
    return parts
 
 
def semantic_chunk_text(text: str, max_chunk_size: int = 512, min_chunk_size: int = 100, overlap_sentences: int = 2) -> List[str]:
    """
    Chunk text by sentences, respecting semantic boundaries.
    Keeps related sentences together for better context.
   
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk (filter out small chunks)
        overlap_sentences: Number of sentences to overlap between chunks
   
    Returns:
        List of text chunks
    """
    from nltk.tokenize import sent_tokenize
   
    # Clean text
    clean_text = re.sub(r"\s+", " ", text).strip()
   
    if not clean_text:
        return []
   
    # Split into sentences
    try:
        sentences = sent_tokenize(clean_text)
    except Exception as e:
        print(f"Sentence tokenization failed, falling back to fixed chunking: {e}", file=sys.stderr)
        return chunk_text(text, chunk_size=max_chunk_size)
   
    if not sentences:
        return []
   
    chunks = []
    current_chunk = []
    current_length = 0
   
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)
       
        # If adding this sentence exceeds max, save current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            if len(chunk_text_str) >= min_chunk_size:
                chunks.append(chunk_text_str)
           
            # Keep last N sentences for context overlap
            if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_length = 0
       
        current_chunk.append(sentence)
        current_length += sentence_length
        i += 1
   
    # Add remaining chunk
    if current_chunk:
        chunk_text_str = ' '.join(current_chunk)
        if len(chunk_text_str) >= min_chunk_size:
            chunks.append(chunk_text_str)
   
    return chunks if chunks else [clean_text]  # Return original if chunking failed
 
def embed_chunks(chunks: List[str]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    arr = np.array(model.encode(chunks, show_progress_bar=False))
    return arr.astype("float32")
 
def uuid_to_faiss_int(uid: str) -> int:
    return int(uid.replace('-', ''), 16) % (2**63 - 1)
 
# atomic write for metadata
def write_json_atomic(obj: Any, path: str):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
 
def safe_read_metadata(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        if os.path.getsize(path) == 0:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # If corrupt, back it up and return empty
        try:
            bak = f"{path}.corrupt.{int(time.time())}.bak"
            os.replace(path, bak)
            print(f"Backed up corrupt metadata to {bak}", file=sys.stderr)
        except Exception:
            pass
        return {}
 
def save_metadata_list(metadata_list: List[dict], metadata_file: str):
    existing = safe_read_metadata(metadata_file)
    for item in metadata_list:
        existing[item["id"]] = item
    write_json_atomic(existing, metadata_file)
 
# FAISS store helpers
def store_in_faiss(embeddings: np.ndarray, ids: List[str], index_file: str):
    if embeddings is None or len(embeddings) == 0:
        return
    dim = embeddings.shape[1]
    if os.path.exists(index_file):
        try:
            index = faiss.read_index(index_file)
        except Exception as e:
            # if read failed, try to move corrupt index aside and create new
            corrupt = index_file + f".corrupt.{int(time.time())}.bak"
            try:
                os.replace(index_file, corrupt)
                print(f"Backed up corrupt index to {corrupt}", file=sys.stderr)
            except Exception:
                pass
            index = None
    else:
        index = None
 
    if index is None:
        base = faiss.IndexFlatL2(dim)
        id_index = faiss.IndexIDMap(base)
    else:
        id_index = faiss.IndexIDMap(index) if not isinstance(index, faiss.IndexIDMap) else index
 
    int_ids = np.array([uuid_to_faiss_int(uid) for uid in ids], dtype=np.int64)
    embeddings = embeddings.astype("float32")
    id_index.add_with_ids(embeddings, int_ids)
    faiss.write_index(id_index, index_file)
 
def delete_ids_from_faiss(uuid_list: List[str], index_file: str) -> int:
    if not uuid_list or not os.path.exists(index_file):
        return 0
    int_ids = np.array([uuid_to_faiss_int(uid) for uid in uuid_list], dtype=np.int64)
    index = faiss.read_index(index_file)
    id_index = faiss.IndexIDMap(index) if not isinstance(index, faiss.IndexIDMap) else index
    try:
        id_index.remove_ids(int_ids)
        faiss.write_index(id_index, index_file)
        return len(int_ids)
    except Exception as e:
        print("Failed to remove ids:", e, file=sys.stderr)
        return 0
 
def delete_metadata_by_ids(ids: List[str], metadata_file: str) -> int:
    existing = safe_read_metadata(metadata_file)
    removed = 0
    for uid in ids:
        if uid in existing:
            del existing[uid]
            removed += 1
    write_json_atomic(existing, metadata_file)
    return removed
 
def delete_by_filename(file_name: str, index_file: str, metadata_file: str) -> dict:
    if not os.path.exists(metadata_file):
        return {"found": 0, "faiss_removed": 0, "metadata_removed": 0}
    existing = safe_read_metadata(metadata_file)
    ids = [uid for uid, item in existing.items() if item.get("fileName") == file_name or item.get("file_name") == file_name]
    found = len(ids)
    faiss_removed = delete_ids_from_faiss(ids, index_file) if found else 0
    metadata_removed = delete_metadata_by_ids(ids, metadata_file) if found else 0
    return {"found": found, "faiss_removed": faiss_removed, "metadata_removed": metadata_removed}
 
# -----------------------------
# PDF -> OCR JSON helpers
# -----------------------------
def pdf_to_json_list(pdf_bytes: bytes, file_name: str, source_link: str, location: str, dpi: int = DEFAULT_DPI, ocr_lang: str = OCR_LANG):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_list = []
   
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_json = {"fileName": file_name, "sourceLink": source_link or "", "location": location, "page": page_index + 1, "lines": []}
       
        try:
            # OPTIMIZATION: Check if page has embedded text first (much faster than OCR)
            text = page.get_text()
           
            # If page has sufficient text (>50 chars), use it directly - skip slow OCR
            if text and len(text.strip()) > 50:
                print(f"Page {page_index + 1} has embedded text, skipping OCR", file=sys.stderr)
                page_json["lines"].append({
                    "text": text.strip(),
                    "bbox": {"x0": 0, "y0": 0, "x1": int(page.rect.width), "y1": int(page.rect.height)},
                    "confidence": 1.0  # High confidence for embedded text
                })
                pages_list.append(page_json)
                continue
           
            # If no embedded text or very little, try OCR (for scanned PDFs/images)
            if not EASYOCR_AVAILABLE or ocr_reader is None:
                print(f"Page {page_index + 1} has no embedded text, but EasyOCR not available", file=sys.stderr)
                if text.strip():
                    page_json["lines"].append({
                        "text": text.strip(),
                        "bbox": {"x0": 0, "y0": 0, "x1": int(page.rect.width), "y1": int(page.rect.height)},
                        "confidence": 0.3
                    })
                pages_list.append(page_json)
                continue
           
            print(f"Page {page_index + 1} has no embedded text, attempting EasyOCR", file=sys.stderr)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
           
            # Convert PIL Image to numpy array for EasyOCR
            img_array = np.array(img)
           
            # Use EasyOCR (pure Python, no external dependencies)
            try:
                results = ocr_reader.readtext(img_array)
               
                # EasyOCR returns: [(bbox, text, confidence), ...]
                for bbox, txt, conf in results:
                    if not txt.strip():
                        continue
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    page_json["lines"].append({
                        "text": txt.strip(),
                        "bbox": {
                            "x0": int(min(x_coords)),
                            "y0": int(min(y_coords)),
                            "x1": int(max(x_coords)),
                            "y1": int(max(y_coords))
                        },
                        "confidence": float(conf)
                    })
                   
            except Exception as ocr_error:
                # If EasyOCR fails, fall back to whatever text we can get
                print(f"EasyOCR failed for page {page_index + 1}, using available text: {str(ocr_error)[:100]}", file=sys.stderr)
                if text.strip():
                    page_json["lines"].append({
                        "text": text.strip(),
                        "bbox": {"x0": 0, "y0": 0, "x1": int(page.rect.width), "y1": int(page.rect.height)},
                        "confidence": 0.5
                    })
                pages_list.append(page_json)
                continue
               
        except Exception as page_error:
            # If entire page processing fails, try text extraction as last resort
            print(f"Page {page_index + 1} processing failed, attempting text extraction: {str(page_error)[:100]}", file=sys.stderr)
            try:
                text = page.get_text()
                if text.strip():
                    page_json["lines"].append({
                        "text": text.strip(),
                        "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
                        "confidence": 0.0
                    })
            except Exception as text_error:
                print(f"Text extraction also failed for page {page_index + 1}: {str(text_error)[:100]}", file=sys.stderr)
       
        pages_list.append(page_json)
   
    doc.close()
    return pages_list
 
def extract_lines_with_pages(data: List[dict]) -> List[Tuple[str, int]]:
    results = []
    for page in data:
        page_number = page.get("page", 0)
        lines = page.get("lines", [])
        sorted_lines = sorted(lines, key=lambda l: (l["bbox"]["y0"], l["bbox"]["x0"]))
        page_text = ""
        last_y = None
        for line in sorted_lines:
            text = line.get("text", "").strip()
            if not text:
                continue
            if last_y is not None and abs(line["bbox"]["y0"] - last_y) > 10:
                page_text += "\n"
            page_text += (" " + text if page_text else text)
            last_y = line["bbox"]["y0"]
        if page_text.strip():
            results.append((page_text, page_number))
    return results
 
# -----------------------------
# JSON extraction helper (generic)
# -----------------------------
def extract_texts_from_json(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    extracted: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            extracted.extend(extract_texts_from_json(v, new_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            extracted.extend(extract_texts_from_json(item, new_path))
    elif isinstance(obj, str):
        if obj.strip():
            extracted.append((obj, path))
    return extracted
 
# -----------------------------
# Excel helpers (unchanged)
# -----------------------------
def is_excel(filename: str) -> bool:
    excel_exts = [".xls", ".xlsx", ".csv"]
    return any(filename.lower().endswith(ext) for ext in excel_exts)
 
def excel_to_records(file_bytes: bytes, file_name: str, location: str, source_link: str):
    ext = file_name.lower().split('.')[-1]
    df_dict = {}
    try:
        if ext in ["xlsx", "xls"]:
            # openpyxl for xlsx; xlrd for xls (xlrd no longer supports xlsx in recent versions)
            df_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        elif ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='ISO-8859-1')
            df_dict = {"Sheet1": df}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, XLSX, XLS, or CSV.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel/CSV file: {str(e)}")
    records = []
    for sheet_name, df in df_dict.items():
        df = df.fillna("")
        for idx, row in df.iterrows():
            row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            records.append({"text": row_text, "sheet": sheet_name, "row_number": idx + 1, "fileName": file_name, "location": location, "sourceLink": source_link})
    return records
 
# -----------------------------
# Request/Response models
# -----------------------------
class ScrapeRequest(BaseModel):
    url: str
    location: str
    max_pages: Optional[int] = DEFAULT_MAX_PAGES
    recursive: Optional[bool] = False

class ProcessResponse(BaseModel):
    chunks_stored: int
    text_fields_found: int
    location: str
    fileName: str
    sourceLink: Optional[str] = None
    metadata_path: str
    index_path: str
    pages_count: int
    ocr_preview_first_page: Optional[str] = None

def determine_audience(text: str) -> str:
    """
    Determine audience based on keywords.
    """
    text_lower = text.lower()
    # Keywords that imply the content is targeting members or potential members
    member_keywords = ["member", "membership", "join", "renew", "login", "exclusive", "journal", "sig", "special interest group"]
    if any(kw in text_lower for kw in member_keywords):
        return "member"
    return "anonymous"

def ingest_text_file(file_bytes: bytes, file_name: str, location: str, source_link: str = None) -> dict:
    """
    Ingest a text/markdown file.
    
    Splits content by headers (#, ##, ###) to determine 'topic'.
    Chunk text within sections.
    """
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = file_bytes.decode("latin-1")  # Fallback
        
    lines = content.splitlines()
    
    current_topic = "General"
    current_section_lines = []
    
    sections = [] # List of (topic, text)
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # New section start
            # Save previous section if it has content
            if current_section_lines:
                sections.append((current_topic, "\n".join(current_section_lines)))
                current_section_lines = []
            
            # Update topic (remove # and whitespace)
            current_topic = stripped.lstrip("#").strip()
        else:
            current_section_lines.append(line)
            
    # Add the last section
    if current_section_lines:
        sections.append((current_topic, "\n".join(current_section_lines)))
        
    index_file, metadata_file = get_index_and_metadata_paths(location)
    
    # delete previous entries for same fileName
    try:
        delete_by_filename(file_name, index_file, metadata_file)
    except Exception as e:
        print(f"Warning: Failed to delete old entries for {file_name}: {e}", file=sys.stderr)
    
    text_chunks = []
    chunk_ids = []
    metadata_list = []
    
    site_url = "https://aasldv2022dev.aasld.org/" # Hardcoded as per requirements
    
    for topic, text in sections:
        if not text.strip():
            continue
            
        audience = determine_audience(text) # Or determine audience based on topic? Using text for now gives more context.
        # Alternatively, check topic for 'member' keywords too
        if determine_audience(topic) == "member":
            audience = "member"

        chunks = semantic_chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        for cid, chunk in zip(ids, chunks):
            metadata_list.append({
                "id": cid,
                "text": chunk,
                "fileName": file_name,
                "location": location,
                "site": site_url,
                "sourceLink": source_link or site_url,
                "topic": topic,
                "audience": audience,
                "contentType": "text"
            })
            
        text_chunks.extend(chunks)
        chunk_ids.extend(ids)
        
    if not text_chunks:
         # Still return valid response structure even if empty
         return {
            "chunks_stored": 0,
            "text_fields_found": 0,
            "location": location,
            "fileName": file_name,
            "sourceLink": source_link,
            "metadata_path": metadata_file,
            "index_path": index_file,
            "pages_count": 0,
            "ocr_preview_first_page": None
        }

    embeddings = embed_chunks(text_chunks)
    store_in_faiss(embeddings, chunk_ids, index_file)
    save_metadata_list(metadata_list, metadata_file)
    
    return {
        "chunks_stored": len(text_chunks),
        "text_fields_found": len(metadata_list),
        "location": location,
        "fileName": file_name,
        "sourceLink": source_link,
        "metadata_path": metadata_file,
        "index_path": index_file,
        "pages_count": 0, # Not applicable for text
        "ocr_preview_first_page": (text_chunks[0][:200] if text_chunks else None)
    }
 
# -----------------------------
# Direct Ingestion Function (Reusable)
# -----------------------------
def ingest_json_body(json_payload: Any, file_name: str, location: str, source_link: str = None) -> dict:
    # file_name and location already required
    # We'll store JSON-derived content into a separate index/metadata path
    index_file, metadata_file = get_index_and_metadata_paths_for_json(location)
 
    # If the payload matches your sample structure (content_html + weburl + fileName),
    # prefer those fields.
    inner_file_name = file_name
    inner_source = source_link or ""
    texts = []
 
    if isinstance(json_payload, dict) and "content_html" in json_payload:
        # Prefer weburl -> sourceLink
        inner_file_name = json_payload.get("fileName") or json_payload.get("file_name") or file_name
        inner_source = json_payload.get("weburl") or json_payload.get("sourceLink") or inner_source or ""
        # extract text from content_html (strip tags)
        html = json_payload.get("content_html", "")
        # rudimentary HTML->text (keeps line breaks where <p> or <br> exist)
        # replace <br> and variants with newline, then strip tags
        html = re.sub(r'(?i)<br\s*/?>', '\n', html)
        html = re.sub(r'(?i)</p>', '\n', html)
        text_content = re.sub(r'<[^>]+>', '', html)  # naive but effective for many cases
        if text_content.strip():
            texts.append(("content_html", text_content.strip()))
    else:
        # fallback: recursively extract all strings from the JSON
        texts = extract_texts_from_json(json_payload)
 
        # attempt to find inner file/source if present
        def find_key(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    r = find_key(v, key)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for it in obj:
                    r = find_key(it, key)
                    if r is not None:
                        return r
            return None
        inner_file_name = find_key(json_payload, "fileName") or find_key(json_payload, "file_name") or file_name
        inner_source = find_key(json_payload, "weburl") or find_key(json_payload, "sourceLink") or inner_source or ""
 
    # delete previous entries for same fileName in json index
    try:
        deletion_result = delete_by_filename(inner_file_name, index_file, metadata_file)
    except Exception as e:
        deletion_result = {"ok": False, "error": str(e)}
 
    all_chunks = []
    all_ids = []
    metadata_list = []
 
    for text_item in texts:
        # text_item might be (text, path) or key,value — handle both shapes
        if isinstance(text_item, tuple) and len(text_item) == 2:
            text_str = text_item[0] if isinstance(text_item[0], str) and len(text_item[0]) > len(text_item[1]) else text_item[1]
            # prefer the longer entry if one side is path
            text_str = text_str.strip()
            path = text_item[1]
        else:
            text_str = str(text_item).strip()
            path = ""
 
        if not text_str:
            continue
 
        chunks = semantic_chunk_text(text_str)
        if not chunks:
            continue
        ids = [str(uuid.uuid4()) for _ in chunks]
        all_chunks.extend(chunks)
        all_ids.extend(ids)
        for cid, chunk in zip(ids, chunks):
            metadata_list.append({
                "id": cid,
                "text": chunk,
                "site": "",
                # "sourceLink": inner_source,
                "audience": "",
                "contentType":"",
                # "direct_link": f"{inner_source}" if inner_source else ""
            })
 
    if not all_chunks:
        # still save metadata (maybe empty)
        save_metadata_list(metadata_list, metadata_file)
        return {
            "chunks_stored": 0,
            "text_fields_found": len(metadata_list),
            "location": location,
            "fileName": inner_file_name,
            "sourceLink": inner_source,
            "metadata_path": metadata_file,
            "index_path": index_file,
            "pages_count": 0,
            "ocr_preview_first_page": None
        }
    # embed + store
    embeddings = embed_chunks(all_chunks)
    store_in_faiss(embeddings, all_ids, index_file)
    save_metadata_list(metadata_list, metadata_file)
    return {
        "chunks_stored": len(all_chunks),
        "text_fields_found": len(metadata_list),
        "location": location,
        "fileName": inner_file_name,
        "sourceLink": inner_source,
        "metadata_path": metadata_file,
        "index_path": index_file,
        "pages_count": 0,
        "ocr_preview_first_page": (all_chunks[0][:200] if all_chunks else None)
    }
 
# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}
 
@app.post("/process", response_model=ProcessResponse)
async def process_pdf(
    request: Request,
    file: UploadFile = File(None),
    pdf_url: str = Form(None),
    file_name: str = Form(...),
    source_link: str = Form(None),
    location: str = Form(...),
    dpi: int = Form(DEFAULT_DPI),
    ocr_lang: str = Form(OCR_LANG),
):
    """
    Accepts either:
     - file upload (multipart/form-data) as `file` (pdf, json, excel)
     - remote URL as `pdf_url`
    Required form fields: file_name, location
    """
    # Read incoming bytes or JSON
    file_bytes: Optional[bytes] = None
    incoming_is_json = False
    json_payload = None
 
    if file and file.filename:
        content_type = (file.content_type or "").lower()
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        # detect JSON content (either content-type or extension)
        if content_type in ("application/json", "text/json") or file.filename.lower().endswith(".json"):
            incoming_is_json = True
            try:
                json_payload = json.loads(raw.decode("utf-8"))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse uploaded JSON: {e}")
        else:
            file_bytes = raw
    elif pdf_url:
        try:
            resp = requests.get(pdf_url, timeout=60, verify=False)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            
            if "text/html" in content_type:
                # Handle general URL scraping
                scraper = WebScraper(pdf_url)
                data = scraper.get_page_content(pdf_url)
                if data:
                    # Treat as if it's a text file but keep title
                    text_chunks = semantic_chunk_text(data["text"])
                    chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
                    metadata_list = []
                    for cid, chunk in zip(chunk_ids, text_chunks):
                        metadata_list.append({
                            "id": cid,
                            "text": chunk,
                            "fileName": data["title"],
                            "location": location,
                            "sourceLink": pdf_url,
                            "topic": data["title"]
                        })
                    index_file, metadata_file = get_index_and_metadata_paths(location)
                    # delete previous entries for same fileName
                    try:
                        delete_by_filename(data["title"], index_file, metadata_file)
                    except Exception as e:
                        print(f"Warning: Failed to delete old entries for {data['title']}: {e}", file=sys.stderr)

                    embeddings = embed_chunks(text_chunks)
                    store_in_faiss(embeddings, chunk_ids, index_file)
                    save_metadata_list(metadata_list, metadata_file)
                    return {
                        "chunks_stored": len(text_chunks),
                        "text_fields_found": len(metadata_list),
                        "location": location,
                        "fileName": data["title"],
                        "sourceLink": pdf_url,
                        "metadata_path": metadata_file,
                        "index_path": index_file,
                        "pages_count": 0,
                        "ocr_preview_first_page": (text_chunks[0][:200] if text_chunks else None)
                    }
                else:
                    raise HTTPException(status_code=400, detail="Failed to scrape text from URL.")
            else:
                # Handle as binary (PDF, etc.)
                file_bytes = resp.content
                if not source_link:
                    source_link = pdf_url
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process URL: {e}")
    else:
        # Maybe user sent raw JSON body (application/json) directly to this endpoint
        content_type_hdr = request.headers.get("content-type", "").lower()
        if "application/json" in content_type_hdr:
            try:
                json_payload = await request.json()
                incoming_is_json = True
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse JSON body: {e}")
        else:
            raise HTTPException(status_code=400, detail="Provide either a file upload, pdf_url, or JSON body.")
 
    # If incoming JSON => process into separate JSON index
    if incoming_is_json:
        return ingest_json_body(json_payload, file_name, location, source_link)
 
    # If we get here, it's a binary file or text file
    
    # Check for Text file
    if file_name.lower().endswith(".txt") or file_name.lower().endswith(".md"):
         return ingest_text_file(file_bytes, file_name, location, source_link)
         
    if is_excel(file_name):
        # Excel ingestion path
        records = excel_to_records(file_bytes, file_name, location, source_link or "")
        text_chunks = []
        chunk_ids = []
        metadata_list = []
        for record in records:
            chunks = semantic_chunk_text(record["text"])
            ids = [str(uuid.uuid4()) for _ in chunks]
            for cid, chunk in zip(ids, chunks):
                metadata_list.append({
                    "id": cid,
                    "text": chunk,
                    "fileName": file_name,
                    "location": location,
                    "sheet": record["sheet"],
                    "row_number": record["row_number"],
                    "sourceLink": record.get("sourceLink", "")
                })
            text_chunks.extend(chunks)
            chunk_ids.extend(ids)
        index_file, metadata_file = get_index_and_metadata_paths(location)
        # delete previous entries for same fileName
        try:
            delete_by_filename(file_name, index_file, metadata_file)
        except Exception as e:
            print(f"Warning: Failed to delete old entries for {file_name}: {e}", file=sys.stderr)

        embeddings = embed_chunks(text_chunks)
        store_in_faiss(embeddings, chunk_ids, index_file)
        save_metadata_list(metadata_list, metadata_file)
        return {
            "chunks_stored": len(text_chunks),
            "text_fields_found": len(metadata_list),
            "location": location,
            "fileName": file_name,
            "sourceLink": source_link,
            "metadata_path": metadata_file,
            "index_path": index_file,
            "pages_count": 0,
            "ocr_preview_first_page": (text_chunks[0][:200] if text_chunks else None)
        }
 
    # PDF path
    try:
        ocr_json = pdf_to_json_list(pdf_bytes=file_bytes, file_name=file_name, source_link=source_link or "", location=location, dpi=dpi, ocr_lang=ocr_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
 
    def merge_bboxes(lines):
        xs0 = [l["bbox"]["x0"] for l in lines] if lines else []
        ys0 = [l["bbox"]["y0"] for l in lines] if lines else []
        xs1 = [l["bbox"]["x1"] for l in lines] if lines else []
        ys1 = [l["bbox"]["y1"] for l in lines] if lines else []
        return {"x0": min(xs0) if xs0 else None, "y0": min(ys0) if ys0 else None, "x1": max(xs1) if xs1 else None, "y1": max(ys1) if ys1 else None}
 
    extracted = extract_lines_with_pages(ocr_json)
    text_chunks = []
    chunk_ids = []
    metadata_list = []
    for text, page_number in extracted:
        chunks = semantic_chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        for cid, chunk in zip(ids, chunks):
            metadata_list.append({
                "id": cid,
                "text": chunk,
                "fileName": file_name,
                "location": location,
                "page_number": page_number,
                "sourceLink": source_link or "",
                "direct_link": f"{(source_link or '')}#page={page_number}" if source_link else f"#page={page_number}",
                "bbox": merge_bboxes(ocr_json[page_number - 1]["lines"]) if page_number - 1 < len(ocr_json) else None
            })
        text_chunks.extend(chunks)
        chunk_ids.extend(ids)
 
    index_file, metadata_file = get_index_and_metadata_paths(location)
    # delete previous entries for same fileName
    try:
        delete_by_filename(file_name, index_file, metadata_file)
    except Exception as e:
        print(f"Warning: Failed to delete old entries for {file_name}: {e}", file=sys.stderr)

    if not text_chunks:
        save_metadata_list(metadata_list, metadata_file)
        return JSONResponse({
            "chunks_stored": 0,
            "text_fields_found": len(metadata_list),
            "location": location,
            "fileName": file_name,
            "sourceLink": source_link,
            "metadata_path": metadata_file,
            "index_path": index_file,
            "pages_count": len(ocr_json),
            "ocr_preview_first_page": (extracted[0][0][:200] if extracted else None)
        })
 
    try:
        embeddings = embed_chunks(text_chunks)
        store_in_faiss(embeddings, chunk_ids, index_file)
        save_metadata_list(metadata_list, metadata_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
 
    return {
        "chunks_stored": len(text_chunks),
        "text_fields_found": len(metadata_list),
        "location": location,
        "fileName": file_name,
        "sourceLink": source_link,
        "metadata_path": metadata_file,
        "index_path": index_file,
        "pages_count": len(ocr_json),
        "ocr_preview_first_page": (extracted[0][0][:200] if extracted else None)
    }

def run_scrape_task(request: ScrapeRequest):
    """
    Background worker for scraping and ingestion.
    """
    try:
        scraper = WebScraper(request.url)
        
        if request.recursive:
            pages = scraper.crawl(max_pages=request.max_pages)
        else:
            content = scraper.get_page_content(request.url)
            pages = [content] if content else []
        
        if not pages:
            print(f"Scrape Task Failed: No content found for {request.url}")
            return
        
        total_chunks = 0
        all_metadata = []
        text_chunks_accumulated = []
        ids_accumulated = []
        
        index_file, metadata_file = get_index_and_metadata_paths(request.location)
        
        url_to_info = {p["url"]: {"title": p["title"], "snippet": p.get("snippet", "")} for p in pages}
        
        for page in pages:
            # Enrich outgoing links with target page info from our crawl if available
            enriched_links = []
            for link in page.get("outgoing_links", []):
                link_copy = link.copy()
                if link["url"] in url_to_info:
                    target = url_to_info[link["url"]]
                    link_copy["target_title"] = target["title"]
                    link_copy["target_snippet"] = target["snippet"]
                enriched_links.append(link_copy)

            chunks = semantic_chunk_text(page["text"])
            uids = [str(uuid.uuid4()) for _ in chunks]
            
            for cid, chunk in zip(uids, chunks):
                all_metadata.append({
                    "id": cid,
                    "text": chunk,
                    "fileName": page["title"],
                    "location": request.location,
                    "sourceLink": page["url"],
                    "topic": page["title"]
                })
            
            text_chunks_accumulated.extend(chunks)
            ids_accumulated.extend(uids)
            total_chunks += len(chunks)
            
        if text_chunks_accumulated:
            embeddings = embed_chunks(text_chunks_accumulated)
            store_in_faiss(embeddings, ids_accumulated, index_file)
            save_metadata_list(all_metadata, metadata_file)
            print(f"Scrape Task Completed: {len(pages)} pages, {total_chunks} chunks stored at {request.location}")
            
    except Exception as e:
        print(f"Scrape Task Error: {e}")

@app.post("/scrape")
async def scrape_site(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Starts a website scraping and ingestion process in the background.
    """
    background_tasks.add_task(run_scrape_task, request)
    return {
        "status": "started",
        "message": f"Scraping of {request.url} has started in the background. Content will be stored at '{request.location}'.",
        "location": request.location
    }