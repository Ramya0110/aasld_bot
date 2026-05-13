import streamlit as st
import httpx
import json
import pandas as pd
import time
import os

# Configuration
INGESTION_SERVICE_URL = "http://localhost:8001"
RETRIEVAL_SERVICE_URL = "http://localhost:8000"
DEFAULT_LOCATION = "aasld_full_site"
TOP_K = 10
INCLUDE_SUMMARY = True

st.set_page_config(
    page_title="AASLD Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for common settings
with st.sidebar:
    st.title("⚙️ Assistant Settings")
    st.markdown("Assistant settings are now fixed for optimal performance.")
    
    st.divider()
    st.header("🛠️ Service Health")
    if st.button("Check Status"):
        try:
            # Retrieval search as proxy
            retrieval_search = httpx.post(f"{RETRIEVAL_SERVICE_URL}/search", json={"query": "ping", "location": "ping"}, timeout=5)
            st.write(f"Retrieval API: {'🟢 Online' if retrieval_search.status_code < 500 else '🔴 Offline'}")
        except:
            st.write("❌ Error connecting to retrieval service.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Application: Chat Assistant ---
st.markdown("### 🤖 AASLD Knowledge Assistant")
st.markdown("Ask questions about AASLD guidelines, publications, and resources.")

# Container for scrollable chat history
chat_container = st.container(height=600, border=False)

with chat_container:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input (Stay fixed at the bottom of the tab/screen)
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately in the container
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate Assistant Response
    with chat_container:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and thinking..."):
                    payload = {
                        "query": prompt,
                        "location": DEFAULT_LOCATION,
                        "top_k": TOP_K,
                        "include_summary": INCLUDE_SUMMARY
                    }
                    response = httpx.post(f"{RETRIEVAL_SERVICE_URL}/search", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        full_content = ""
                        if data.get("summary"):
                            full_content = data["summary"]
                            st.markdown(full_content)
                        else:
                            if data.get("results"):
                                full_content = "I found some relevant information. Please check the source references below."
                            else:
                                full_content = "No relevant results found for your query. Try a different question or location tag."
                            st.write(full_content)
                        
                        
                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_content
                        })
                        
                        # Force a rerun to clean up the UI and sync history properly
                        st.rerun()
                    else:
                        error_msg = f"❌ Error from retrieval service: {response.text}"
                        st.error(error_msg)
            except Exception as e:
                error_msg = f"❌ Connection Error: {e}"
                st.error(error_msg)
