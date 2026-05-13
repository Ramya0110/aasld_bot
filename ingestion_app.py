import streamlit as st
import httpx
import json
import pandas as pd
import time
import os

# Configuration
INGESTION_SERVICE_URL = "http://localhost:8001"
DEFAULT_LOCATION = "aasld_general"

st.set_page_config(
    page_title="AASLD Ingestion Studio",
    page_icon="📥",
    layout="wide"
)

# Sidebar for common settings
with st.sidebar:
    st.title("⚙️ Ingestion Settings")
    st.markdown("Ingestion settings are now fixed for optimal performance.")
    
    st.divider()
    st.header("🛠️ Service Health")
    if st.button("Check Status"):
        try:
            ingest_health = httpx.get(f"{INGESTION_SERVICE_URL}/health", timeout=5)
            st.write(f"Ingestion API: {'🟢 Online' if ingest_health.status_code == 200 else '🔴 Offline'}")
        except:
            st.write("❌ Error connecting to ingestion service.")

# Main Application Tabs
tabs = st.tabs(["🌐 Web Crawler", "📁 Data Ingestion"])

# --- Tab 1: Web Crawler ---
with tabs[0]:
    st.header("🌐 Website Crawler & Scraper")
    st.markdown(f"Trigger a background process to ingest content into `{DEFAULT_LOCATION}`.")
    
    with st.form("scrape_form"):
        url = st.text_input("Website URL", value="https://aasldv2022dev.aasld.org/")
        max_pages = st.number_input("Max Pages", min_value=1, max_value=500, value=10)
        recursive = st.checkbox("Recursive Scraping", value=True)
        
        if st.form_submit_button("Submit Crawl Task"):
            try:
                payload = {"url": url, "location": DEFAULT_LOCATION, "max_pages": max_pages, "recursive": recursive}
                resp = httpx.post(f"{INGESTION_SERVICE_URL}/scrape", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.success(f"✅ {resp.json()['message']}")
                else:
                    st.error(f"❌ Failed: {resp.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Tab 2: Data Ingestion ---
with tabs[1]:
    st.header("📁 Document Ingestion")
    st.markdown(f"Upload local files for instant processing into `{DEFAULT_LOCATION}`.")
    
    uploaded_file = st.file_uploader("Upload File (PDF, TXT, Excel)", type=["pdf", "txt", "md", "xlsx"])
    
    if st.button("Ingest Document"):
        if uploaded_file:
            try:
                with st.spinner("Processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"location": DEFAULT_LOCATION}
                    resp = httpx.post(f"{INGESTION_SERVICE_URL}/process", files=files, data=data, timeout=120)
                    if resp.status_code == 200:
                        st.success(f"✅ Success! Ingested {resp.json()['chunks_stored']} chunks.")
                    else:
                        st.error(f"❌ Failed: {resp.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please provide a file.")
