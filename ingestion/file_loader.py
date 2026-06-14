"""
Module: ingestion/file_loader.py
Description: Loads and processes general documents (PDFs, Word, TXT, web content).
             Extracts text while preserving metadata for semantic chunking.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 1. TEXT EXTRACTION BY FILE TYPE
# ==========================================

class TextExtractor:
    """Handles text extraction from different document formats."""
    
    @staticmethod
    def extract_pdf(file_path: Path) -> str:
        """Extract text from PDF using PyPDF2 or pdfplumber."""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            logger.warning("PyPDF2 not installed. Trying pdfplumber...")
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                raise ImportError(
                    "No PDF library available. Install: pip install PyPDF2 pdfplumber"
                )

    @staticmethod
    def extract_docx(file_path: Path) -> str:
        """Extract text from Word documents."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

    @staticmethod
    def extract_txt(file_path: Path) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def extract_markdown(file_path: Path) -> str:
        """Extract text from Markdown files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @classmethod
    def extract(cls, file_path: Path) -> str:
        """Route to appropriate extractor based on file extension."""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return cls.extract_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return cls.extract_docx(file_path)
        elif ext in ['.txt', '.text']:
            return cls.extract_txt(file_path)
        elif ext in ['.md', '.markdown']:
            return cls.extract_markdown(file_path)
        else:
            # Try as plain text
            try:
                return cls.extract_txt(file_path)
            except Exception:
                raise ValueError(f"Unsupported file type: {ext}")


# ==========================================
# 2. DOCUMENT LOADER
# ==========================================

class DocumentFileLoader:
    """
    Loads general documents (PDFs, Word, TXT, Markdown) from a directory.
    Extracts text and preserves file metadata.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt', '.text', '.md', '.markdown'
    }
    
    def __init__(self, data_dir: str = "data", recursive: bool = True):
        """
        Initialize document loader.
        
        Args:
            data_dir: Path to directory containing documents
            recursive: If True, scan subdirectories
        """
        self.data_dir = Path(data_dir)
        self.recursive = recursive
        self.max_file_size_mb = 50
        self.extractor = TextExtractor()
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            logger.info(f"Creating empty directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_files(self) -> List[Dict[str, Any]]:
        """
        Scan data directory and load all supported document files.
        
        Returns:
            List of document metadata dictionaries with keys:
            - file_path: absolute path to file
            - file_name: basename
            - extension: file extension
            - file_type: detected type (pdf, docx, txt, markdown)
            - content: extracted text content
            - size_bytes: file size
            - word_count: approximate word count
        """
        loaded_documents: List[Dict[str, Any]] = []
        
        # Determine glob pattern
        pattern = "**/*" if self.recursive else "*"
        
        for file_path in self.data_dir.glob(pattern):
            # Skip directories and hidden files
            if file_path.is_dir() or file_path.name.startswith('.'):
                continue
            
            # Check if supported file type
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue
            
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                logger.warning(f"Skipping {file_path} (too large: {size_mb:.1f} MB)")
                continue
            
            try:
                # Extract text content
                content = self.extractor.extract(file_path)
                
                if not content or not content.strip():
                    logger.warning(f"No text extracted from {file_path}")
                    continue
                
                # Build metadata
                file_type = file_path.suffix.lower().strip('.')
                doc_info = {
                    "file_path": str(file_path.absolute()),
                    "file_name": file_path.name,
                    "extension": file_path.suffix,
                    "file_type": file_type,
                    "content": content,
                    "size_bytes": file_path.stat().st_size,
                    "word_count": len(content.split()),
                }
                
                loaded_documents.append(doc_info)
                logger.info(
                    f"Loaded {file_type.upper()}: {file_path.name} "
                    f"({doc_info['word_count']} words)"
                )
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return loaded_documents


# Alias for backward compatibility with main.py
DevOpsFileLoader = DocumentFileLoader
