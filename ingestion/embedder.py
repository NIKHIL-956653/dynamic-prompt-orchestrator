"""
Module: ingestion/embedder.py
Description: Generates embeddings for document chunks and builds FAISS index
             for fast semantic search.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 1. EMBEDDING MODEL HANDLER
# ==========================================

class EmbeddingModel:
    """Wrapper for embedding models (sentence-transformers, etc.)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()

    def _load_model(self):
        """Load embedding model from sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Batch embedding with progress
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


# ==========================================
# 2. FAISS INDEX BUILDER
# ==========================================

class FAISSIndex:
    """Wrapper for FAISS vector index."""
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (e.g., 384 for MiniLM)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Run: pip install faiss-cpu "
                "(or faiss-gpu for GPU support)"
            )
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.metadata = []  # Store chunk metadata
        self.doc_count = 0

    def add_vectors(self, embeddings: List[List[float]], 
                   metadata: List[Dict[str, Any]]):
        """
        Add embeddings and metadata to index.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (one per embedding)
        """
        import numpy as np
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Store metadata
        self.metadata.extend(metadata)
        self.doc_count += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.doc_count}")

    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search index for nearest neighbors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            (distances, indices) - distances and indices of top-k results
        """
        import numpy as np
        
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        
        return distances[0].tolist(), indices[0].tolist()

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Retrieve metadata for a result index."""
        if 0 <= index < len(self.metadata):
            return self.metadata[index]
        return {}

    def save(self, path: str):
        """Save index and metadata to disk."""
        self.faiss.write_index(self.index, path)
        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str):
        """Load index from disk."""
        self.index = self.faiss.read_index(path)
        logger.info(f"Loaded FAISS index from {path}")


# ==========================================
# 3. EMBEDDING PIPELINE
# ==========================================

class DocumentEmbedder:
    """Orchestrates embedding generation and FAISS indexing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 store_dir: str = "vector_store"):
        """
        Initialize embedder.
        
        Args:
            model_name: Embedding model name from HuggingFace
            store_dir: Directory to save FAISS index and metadata
        """
        self.model = EmbeddingModel(model_name)
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.index = None

    def embed_and_store(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for chunks and build FAISS index.
        
        Args:
            chunks: List of chunk dicts (from chunker)
            
        Returns:
            Index metadata (document count, chunk count, etc.)
        """
        if not chunks:
            logger.warning("No chunks provided")
            return {}
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.embed(texts)
        
        # Create FAISS index
        self.index = FAISSIndex(self.model.dimension)
        
        # Add embeddings and metadata
        metadata = [
            {
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "document_type": chunk.get("document_type", "unknown"),
                "content_preview": chunk["content"][:200],
                "token_count": chunk.get("token_count", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ]
        
        self.index.add_vectors(embeddings, metadata)
        
        # Save index and metadata
        self._save_artifacts(metadata)
        
        # Return summary
        summary = {
            "total_chunks": len(chunks),
            "embedding_model": self.model.model_name,
            "embedding_dimension": self.model.dimension,
            "index_size": self.index.doc_count,
            "unique_documents": len(set(c["document_name"] for c in chunks)),
        }
        
        logger.info(f"✅ Indexing complete: {json.dumps(summary, indent=2)}")
        return summary

    def _save_artifacts(self, metadata: List[Dict[str, Any]]):
        """Save FAISS index and metadata to disk."""
        # Save FAISS index
        index_path = self.store_dir / "index.faiss"
        self.index.save(str(index_path))
        
        # Save metadata
        metadata_path = self.store_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save index config
        config_path = self.store_dir / "config.json"
        config = {
            "model_name": self.model.model_name,
            "dimension": self.model.dimension,
            "num_chunks": len(metadata),
            "created_at": str(Path(__file__).stat().st_mtime),
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of result dicts with content and metadata
        """
        if not self.index:
            raise RuntimeError("No index loaded. Run embed_and_store() first.")
        
        # Embed query
        query_embedding = self.model.embed_single(query)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for dist, idx in zip(distances, indices):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                metadata = self.index.get_metadata(idx)
                metadata["similarity_score"] = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append(metadata)
        
        return results


# ==========================================
# 4. RUNTIME VERIFICATION (Smoke Test)
# ==========================================

if __name__ == "__main__":
    print("--- [DOCUMENT EMBEDDER SMOKE TEST] ---\n")
    
    # Create sample chunks
    sample_chunks = [
        {
            "chunk_id": "doc1_0001",
            "document_name": "ml_basics.txt",
            "document_type": "txt",
            "content": "Machine learning is a subset of artificial intelligence that focuses on enabling computer systems to learn from data.",
            "token_count": 25,
            "chunk_index": 0,
            "total_chunks": 5,
            "metadata": {"source_file": "data/ml_basics.txt", "file_type": "txt"}
        },
        {
            "chunk_id": "doc1_0002",
            "document_name": "ml_basics.txt",
            "document_type": "txt",
            "content": "Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data.",
            "token_count": 23,
            "chunk_index": 1,
            "total_chunks": 5,
            "metadata": {"source_file": "data/ml_basics.txt", "file_type": "txt"}
        },
        {
            "chunk_id": "doc1_0003",
            "document_name": "ml_basics.txt",
            "document_type": "txt",
            "content": "Neural networks are inspired by biological neurons and can learn complex patterns in data.",
            "token_count": 16,
            "chunk_index": 2,
            "total_chunks": 5,
            "metadata": {"source_file": "data/ml_basics.txt", "file_type": "txt"}
        },
    ]
    
    try:
        # Initialize embedder
        print("[1/3] Initializing embedder...")
        embedder = DocumentEmbedder(
            model_name="all-MiniLM-L6-v2",
            store_dir="vector_store"
        )
        print(f"✅ Embedder ready. Model: {embedder.model.model_name}, Dim: {embedder.model.dimension}\n")
        
        # Embed and store
        print("[2/3] Embedding chunks...")
        summary = embedder.embed_and_store(sample_chunks)
        print(f"✅ Embedding complete:\n")
        for k, v in summary.items():
            print(f"   {k}: {v}")
        
        # Test search
        print("\n[3/3] Testing search...")
        query = "What is machine learning?"
        results = embedder.search(query, k=2)
        print(f"✅ Search results for '{query}':\n")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] {result['document_name']} (score: {result['similarity_score']:.3f})")
            print(f"       {result['content_preview'][:100]}...\n")
        
        print("✅ Document embedder is working correctly!")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("Make sure sentence-transformers and FAISS are installed:")
        print("  pip install sentence-transformers faiss-cpu")


# Alias for backward compatibility with main.py
DevOpsEmbedder = DocumentEmbedder
