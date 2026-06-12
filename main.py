"""
Module: main.py
Description: Entrypoint for Phases 1 & 2 of the DevOps Log Analyzer.
             Executes the ETL pipeline: Load -> Semantic Chunk -> Embed -> FAISS.
"""

import os
from ingestion.file_loader import DevOpsFileLoader
from ingestion.chunker import chunk_all
from ingestion.embedder import DevOpsEmbedder

def run_ingestion_pipeline():
    print("=== DEVOPS LOG ANALYZER: INGESTION PIPELINE (PHASE 1 & 2) ===")
    
    # ---------------------------------------------------------
    # 1. LOAD RAW FILES
    # ---------------------------------------------------------
    print("\n[1/3] Scanning and loading data directory...")
    loader = DevOpsFileLoader(data_dir="data")
    raw_files = loader.load_files()
    
    if not raw_files:
        print("❌ No valid files found in 'data/' directory. Please add some YAMLs, Jenkinsfiles, or shell scripts and run again.")
        return

    # Adapt dictionary keys to match your custom chunker's expected format
    formatted_files = []
    for f in raw_files:
        ext = f.get("extension", "").strip(".").lower()
        fname = f.get("file_name", "").lower()
        
        # Determine file type for the chunker router
        if fname in ["jenkinsfile", "dockerfile"]:
            ftype = fname
        else:
            ftype = ext

        formatted_files.append({
            "path": f.get("file_path", ""),
            "file_type": ftype,
            "content": f.get("content", "")
        })

    print(f"✅ Successfully loaded {len(formatted_files)} DevOps files.")

    # ---------------------------------------------------------
    # 2. SEMANTIC CHUNKING
    # ---------------------------------------------------------
    print("\n[2/3] Processing semantic boundaries (YAML, Jenkins stages, Shell functions)...")
    chunks = chunk_all(formatted_files)
    
    if not chunks:
        print("❌ No chunks generated. Check your file contents.")
        return
        
    print(f"✅ Generated {len(chunks)} semantic chunks.")

    # ---------------------------------------------------------
    # 3. EMBED & STORE IN FAISS
    # ---------------------------------------------------------
    print("\n[3/3] Generating dense vector embeddings and building FAISS index...")
    embedder = DevOpsEmbedder(model_name="all-MiniLM-L6-v2", store_dir="vector_store")
    embedder.embed_and_store(chunks)

    print("\n=== 🚀 PIPELINE COMPLETE: KNOWLEDGE BASE IS READY ===")

if __name__ == "__main__":
    run_ingestion_pipeline()