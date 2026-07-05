import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os
import sys
import subprocess

st.set_page_config(page_title="PatentLens AI: Research Copilot", layout="wide", page_icon="🌐")

# Secure Mock Strategy function mapping to eliminate NameErrors
def generate_ai_workaround_strategy(user_input, contested_text):
    return "Modify architectural structural points to decoupling layers. Evade direct vocabulary collision matching with priority claims structures."

# --- 1. CORE TWO-STAGE HYBRID PIPELINE ---
@st.cache_resource
def load_pipeline_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "PatentLens_DB") 
    
    if not os.path.exists(db_path):
        with st.spinner("📦 Initial Setup: Executing on-the-fly local vector space building..."):
            try:
                # 🚨 FIX: Force subprocess to use the active virtual environment python executable
                subprocess.run([sys.executable, os.path.join(current_dir, "ingest_patents.py")], check=True)
                st.success("✅ Database compiled successfully!")
            except Exception as e:
                st.error(f"❌ Automation Failed: Could not trigger ingestion script. Error: {e}")
                st.stop()
                
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    chroma_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    
    chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)
    
    try:
        collection = chroma_client.get_collection(
            name="universal_global_patents",
            embedding_function=sentence_transformer_ef
        )
    except Exception as e:
        st.error(f"❌ Collection 'universal_global_patents' not found at {db_path}")
        st.stop()
        
    all_data = collection.get(include=['documents', 'metadatas'])
    corpus = all_data['documents']
    metadatas = all_data['metadatas']
    ids = all_data['ids']
    
    if len(corpus) == 0:
        st.error("❌ Vector store is currently empty.")
        st.stop()
        
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device="cpu")
    
    return collection, bm25, corpus, metadatas, ids, reranker

# Load Assets natively outside loop contexts
collection, bm25, corpus, metadatas, ids, reranker = load_pipeline_assets()

# --- UI HEADER & INPUT INTERFACE ---
st.title("🌐 PatentLens AI: Enterprise Research Copilot")
st.markdown("---")

user_input = st.text_area("✍️ Enter your Technical Concept / Claim Statement:", height=150)
trigger_search = st.button("🚀 Analyze Prior-Art Clashes")

# --- 2. EXECUTION ENGINE FLOW WITH STATE PERSISTENCE ---
if trigger_search and user_input:
    with st.spinner("🧠 Scanning global multi-stage database spaces..."):
        # Run BM25 Lexical Matching
        tokenized_query = user_input.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:15]
        
        # Run Chroma Dense Store Vector Matching
        vector_results = collection.query(
            query_texts=[user_input],
            n_results=15
        )
        vector_ids = vector_results['ids'][0]
        
        # Pool all candidates securely
        candidate_pool = {}
        for idx in top_bm25_indices:
            cid = ids[idx]
            candidate_pool[cid] = {"text": corpus[idx], "metadata": metadatas[idx], "type": "Lexical Match"}
            
        for idx, cid in enumerate(vector_ids):
            if cid not in candidate_pool:
                try:
                    c_idx = ids.index(cid)
                    candidate_pool[cid] = {"text": corpus[c_idx], "metadata": metadatas[c_idx], "type": "Semantic Match"}
                except ValueError:
                    pass

        # Apply Neural Cross-Encoder Re-ranking layer
        pairs = [[user_input, item['text']] for item in candidate_pool.values()]
        rerank_scores = reranker.predict(pairs)
        
        final_sorted_results = []
        for score, (cid, item) in zip(rerank_scores, candidate_pool.items()):
            sim_pct = int(100 / (1 + np.exp(-score)))
            final_sorted_results.append({
                "id": cid,
                "text": item['text'],
                "metadata": item['metadata'],
                "type": item['type'],
                "similarity_pct": max(5, min(98, sim_pct))
            })
            
        final_sorted_results = sorted(final_sorted_results, key=lambda x: x['similarity_pct'], reverse=True)[:5]
        
        # Lock variable in permanent session state proxy
        st.session_state.results_pool = final_sorted_results
        st.session_state.last_query = user_input

# --- 3. WORKSPACE VIEW DISTRIBUTION LAYOUT ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Find Similar Patents", 
    "📊 Compare Ideas (Strategic Cross-Match)", 
    "📄 Generate Research Reports"
])

if "results_pool" in st.session_state and st.session_state.results_pool:
    active_pool = st.session_state.results_pool
    active_query = st.session_state.get('last_query', user_input)
    
    with tab1:
        st.subheader("Targeted Patent Claims Intersections")
        for idx, match in enumerate(active_pool):
            sim_pct = match['similarity_pct']
            
            with st.expander(f"Rank {idx+1} | {match['metadata']['title']} ({match['type']}) — {sim_pct}% Match"):
                col1, col2 = st.columns([1, 4])
                with col1:
                    if sim_pct >= 75:
                        st.error(f"⚠️ HIGH COLLISION\n{sim_pct}% Match")
                    elif sim_pct >= 45:
                        st.warning(f"⚡ MODERATE OVERLAP\n{sim_pct}% Match")
                    else:
                        st.success(f"🟢 LOW RISK\n{sim_pct}% Match")
                        
                with col2:
                    st.markdown(f"**Patent ID:** `{match['metadata']['patent_id']}` | **Filing Date:** `{match['metadata']['date']}`")
                    st.write(match['text'])
                    
    with tab2:
        st.subheader("Strategic Alignment & Collision Matrix")
        for idx, match in enumerate(active_pool):
            sim_pct = match['similarity_pct']
            st.markdown(f"### Priority Node {idx+1}: {match['metadata']['title']}")
            st.progress(int(sim_pct))
            
            ai_advice = generate_ai_workaround_strategy(active_query, match['text'])
            st.info(f"💡 **Strategic Modification Suggestion:**\n{ai_advice}")
            st.divider()
            
    with tab3:
        st.subheader("Export Compliance Manifest")
        report_text = f"======================================================================\n"
        report_text += f"          PATENTLENS AI: SYSTEM PRIOR-ART COMPLIANCE REPORT\n"
        report_text += f"======================================================================\n\n"
        report_text += f"USER CONCEPT EVALUATED:\n\"{active_query}\"\n\n"
        report_text += f"IDENTIFIED CONFLICT VECTORS:\n"
        
        for idx, match in enumerate(active_pool):
            sim_pct = match['similarity_pct']
            ai_advice = generate_ai_workaround_strategy(active_query, match['text'])
            
            report_text += f"[{idx+1}] Rank Vector Analysis:\n"
            report_text += f"  - Document ID: {match['metadata']['patent_id']}\n"
            report_text += f"  - Document Title: {match['metadata']['title']}\n"
            report_text += f"  - System Filing Date: {match['metadata']['date']}\n"
            report_text += f"  - Evaluated Conflict Risk: {sim_pct}%\n"
            report_text += f"  - AI Copilot Strategic Action-Item:\n    {ai_advice}\n"
            report_text += f"  - Contested Chronological Clause Block:\n    \"{match['text']}\"\n\n"
            
        report_text += f"======================================================================\n"
        report_text += f"REPORT GENERATION COMPLETED BY LOCAL TWO-STAGE WORKSTATION PLATFORM\n"
        report_text += f"======================================================================\n"
        
        st.text_area("📜 Compliance Document Manifest Preview", report_text, height=250)
        
        st.download_button(
            label="📥 Download Research Report (.txt Manifest)",
            data=report_text,
            file_name="PatentLens_PriorArt_Compliance_Report.txt",
            mime="text/plain"
        )
else:
    with tab1:
        st.info("💡 Enter your technical concept above and click 'Analyze Prior-Art Clashes' to isolate overlapping claims.")
    with tab2:
        st.info("💡 Enter your technical concept above and click 'Analyze Prior-Art Clashes' to generate mitigation workflows.")
    with tab3:
        st.info("💡 Enter your technical concept above and click 'Analyze Prior-Art Clashes' to compile corporate compliance reports.")