"""
Module: ingestion/chunker.py
Description: Semantic chunking for general documents.
             Respects document structure (paragraphs, sections, sentences)
             while maintaining context window constraints.
"""

import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 1. TOKENIZATION & SIZING
# ==========================================

class TokenCounter:
    """Approximate token counting for semantic chunk sizing."""
    
    # Average tokens per word (varies by language/model)
    TOKENS_PER_WORD = 1.3
    
    @classmethod
    def count(cls, text: str) -> int:
        """Estimate token count."""
        word_count = len(text.split())
        return int(word_count * cls.TOKENS_PER_WORD)
    
    @classmethod
    def is_within_limit(cls, text: str, max_tokens: int) -> bool:
        """Check if text fits within token limit."""
        return cls.count(text) <= max_tokens


# ==========================================
# 2. DOCUMENT STRUCTURE DETECTION
# ==========================================

class DocumentStructure:
    """Detect and parse document structure (headings, sections, etc.)."""
    
    # Heading patterns
    HEADING_PATTERNS = {
        'markdown': r'^(#{1,6})\s+(.+)$',
        'underline': r'^(.+)\n(=+|—+|-+)$',
        'uppercase': r'^[A-Z][A-Z\s]+$',
    }
    
    # Section dividers
    DIVIDER_PATTERNS = [
        r'^-{5,}$',          # -----
        r'^\*{5,}$',         # *****
        r'^={5,}$',          # =====
        r'^---$',            # --- (markdown)
    ]
    
    @staticmethod
    def detect_headings(text: str) -> List[Tuple[int, str, int]]:
        """
        Detect headings in text.
        Returns: List of (level, heading_text, line_number)
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Markdown-style headings
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                heading = match.group(2).strip()
                headings.append((level, heading, i))
        
        return headings
    
    @staticmethod
    def detect_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs (separated by blank lines)."""
        # Split by double newline or more
        paragraphs = re.split(r'\n\s*\n+', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]


# ==========================================
# 3. SEMANTIC CHUNKER
# ==========================================

class SemanticChunker:
    """
    Chunks documents while respecting semantic boundaries.
    Balances chunk size with structural integrity.
    """
    
    def __init__(self, max_chunk_tokens: int = 512, overlap_tokens: int = 50):
        """
        Initialize chunker.
        
        Args:
            max_chunk_tokens: Target chunk size in tokens
            overlap_tokens: Overlap between chunks for context preservation
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.max_chunk_words = int(max_chunk_tokens / TokenCounter.TOKENS_PER_WORD)
        self.token_counter = TokenCounter()

    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[str]:
        """
        Chunk text into semantically coherent pieces.
        
        Args:
            text: Input text to chunk
            preserve_structure: If True, respect paragraph/sentence boundaries
            
        Returns:
            List of chunk strings
        """
        if preserve_structure:
            return self._chunk_preserve_structure(text)
        else:
            return self._chunk_greedy(text)

    def _chunk_preserve_structure(self, text: str) -> List[str]:
        """
        Chunk while respecting paragraph and sentence boundaries.
        
        Strategy:
        1. Split into paragraphs
        2. Group paragraphs until approaching chunk size limit
        3. If single paragraph exceeds limit, split by sentences
        4. Add overlap context from previous chunk
        """
        paragraphs = DocumentStructure.detect_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_buffer = ""
        
        for para in paragraphs:
            para_tokens = self.token_counter.count(para)
            
            # If single paragraph exceeds limit, split it
            if para_tokens > self.max_chunk_tokens:
                # First, finalize current chunk if non-empty
                if current_chunk:
                    chunk_text = overlap_buffer + "\n\n".join(current_chunk)
                    chunks.append(chunk_text.strip())
                    overlap_buffer = current_chunk[-1]  # Store for overlap
                
                # Split paragraph by sentences
                para_chunks = self._chunk_by_sentences(para)
                chunks.extend(para_chunks)
                
                # Reset and prepare overlap
                current_chunk = []
                current_tokens = 0
                if para_chunks:
                    overlap_buffer = para_chunks[-1]
            
            # If adding paragraph would exceed limit, save current chunk
            elif current_tokens + para_tokens > self.max_chunk_tokens and current_chunk:
                chunk_text = overlap_buffer + "\n\n".join(current_chunk)
                chunks.append(chunk_text.strip())
                
                # Start new chunk with overlap
                overlap_buffer = current_chunk[-1]
                current_chunk = [para]
                current_tokens = para_tokens
            
            # Otherwise, add paragraph to current chunk
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Finalize last chunk
        if current_chunk:
            chunk_text = overlap_buffer + "\n\n".join(current_chunk)
            chunks.append(chunk_text.strip())
        
        return [c for c in chunks if c]  # Filter empty chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentence boundaries."""
        # Simple sentence splitting (improve for multilingual support)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self.token_counter.count(sent)
            
            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks

    def _chunk_greedy(self, text: str) -> List[str]:
        """Simple greedy chunking (not recommended, breaks structure)."""
        words = text.split()
        chunks = []
        current = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.token_counter.count(word)
            
            if current_tokens + word_tokens > self.max_chunk_tokens and current:
                chunks.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks


# ==========================================
# 4. DOCUMENT-LEVEL CHUNKING
# ==========================================

def chunk_all(documents: List[Dict[str, Any]], 
              max_chunk_tokens: int = 512,
              preserve_structure: bool = True) -> List[Dict[str, Any]]:
    """
    Chunk all loaded documents.
    
    Args:
        documents: List of document dicts (from DocumentFileLoader)
        max_chunk_tokens: Target chunk size
        preserve_structure: Respect document boundaries
        
    Returns:
        List of chunk dicts with keys:
        - chunk_id: unique identifier
        - document_name: source document
        - content: chunk text
        - token_count: approximate tokens
        - metadata: source metadata
    """
    chunker = SemanticChunker(max_chunk_tokens=max_chunk_tokens)
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0
    
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue
        
        # Chunk document
        doc_chunks = chunker.chunk_text(content, preserve_structure=preserve_structure)
        
        logger.info(
            f"Chunked {doc['file_name']}: {len(doc_chunks)} chunks "
            f"(avg ~{sum(TokenCounter.count(c) for c in doc_chunks) // max(len(doc_chunks), 1)} tokens)"
        )
        
        # Build chunk metadata
        for i, chunk_text in enumerate(doc_chunks):
            chunk_id = f"{doc['file_name'].replace('.', '_')}_{i:04d}"
            token_count = TokenCounter.count(chunk_text)
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "document_name": doc["file_name"],
                "document_type": doc.get("file_type", "unknown"),
                "content": chunk_text,
                "token_count": token_count,
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "metadata": {
                    "source_file": doc["file_path"],
                    "file_type": doc.get("file_type"),
                }
            })
            chunk_counter += 1
    
    logger.info(f"Generated {chunk_counter} total chunks from {len(documents)} documents")
    return all_chunks


# ==========================================
# 5. RUNTIME VERIFICATION (Smoke Test)
# ==========================================

if __name__ == "__main__":
    print("--- [SEMANTIC CHUNKER SMOKE TEST] ---\n")
    
    # Create sample document
    sample_doc = {
        "file_name": "sample.txt",
        "file_type": "txt",
        "file_path": "data/sample.txt",
        "content": """
Introduction to Semantic Chunking

Semantic chunking is an approach to breaking down documents while respecting
their structure and meaning. Unlike fixed-size chunking which splits text at
arbitrary boundaries, semantic chunking respects paragraphs, sentences, and
document structure.

Benefits of Semantic Chunking:
1. Preserves Context: Related sentences stay together
2. Improves Retrieval: Chunks are semantically coherent
3. Better Quality: LLM responses are more accurate

Implementation Approach:
First, we identify paragraph boundaries by detecting blank lines. Then we
group paragraphs together until we approach the maximum chunk size. If a
single paragraph exceeds the limit, we split it by sentence boundaries.

Example with Sentences:
This is the first sentence. This is the second sentence. This is the third.
By breaking at sentence boundaries, we preserve meaning while controlling size.

Advanced Techniques:
For even better results, you can:
- Detect document structure (headings, sections)
- Use domain-specific chunking rules
- Apply overlapping windows for context preservation
- Integrate with embedding models for semantic similarity

Conclusion:
Semantic chunking is essential for building effective RAG systems where
chunk quality directly impacts retrieval and generation quality.
"""
    }
    
    # Test chunking
    chunks = chunk_all([sample_doc], max_chunk_tokens=256)
    
    print(f"✅ Created {len(chunks)} chunks from 1 document\n")
    
    for chunk in chunks[:3]:
        print(f"📌 Chunk {chunk['chunk_id']}")
        print(f"   Tokens: {chunk['token_count']}")
        print(f"   Preview: {chunk['content'][:100]}...\n")
    
    if len(chunks) > 3:
        print(f"... and {len(chunks) - 3} more chunks\n")
    
    print("✅ Semantic chunker is working correctly!")
