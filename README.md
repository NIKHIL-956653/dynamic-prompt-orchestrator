# Dynamic Prompt Orchestrator

A production-grade system for orchestrating LLM prompts with integrated document RAG, multi-backend support, and persistent state management.

## Features

- **Multi-Backend LLM Support**: Groq (free), Ollama (local), Anthropic (cloud), or mock
- **Document RAG Pipeline**: Load PDFs/Word/TXT → Semantic chunking → FAISS indexing
- **Dynamic Prompt Construction**: Jira tickets + product history + user context → optimized prompts
- **Persistent State Management**: Local JSON-based memory for decision tracking
- **Type-Safe**: Pydantic models throughout for validation
- **Modular Architecture**: Independent ingestion and orchestration pipelines

## Quick Start

### Installation

```bash
# Clone or navigate to project directory
cd dynamic-prompt-orchestrator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key (for Groq - free and recommended)
export GROQ_API_KEY=gsk_...  # Get free key from console.groq.com
```

### 1. Build Your Knowledge Base (Optional)

```bash
# Place documents in data/ directory
mkdir -p data/
# Add your PDFs, Word files, TXT, or Markdown files to data/

# Run ingestion pipeline
python main.py

# Output:
# ✓ vector_store/index.faiss (FAISS index)
# ✓ vector_store/metadata.json (chunk metadata)
# ✓ vector_store/config.json (configuration)
```

### 2. Run Prompt Orchestration

```python
from context_manager import ContextPipelineCoordinator
from prompt_orchestrator import UserContext

# Initialize coordinator (defaults to "mock" backend)
coordinator = ContextPipelineCoordinator(provider="groq")

# Simulate Jira webhook payload
jira_ticket = {
    "id": "ENG-707",
    "summary": "Implement Redis Session Cache Cluster",
    "description": "Configure multi-node replication setup...",
    "criteria": [
        "Must replicate session states across 3 nodes",
        "Encryption in transit via TLS required"
    ]
}

# User context
user = UserContext(
    user_role="Senior DevOps Architect",
    preferences={"security_strictness": "Maximum"}
)

# Execute pipeline
response = coordinator.run_ticket_pipeline(jira_ticket, user)

print(f"Answer: {response.final_output}")
print(f"Latency: {response.metrics.execution_time_seconds}s")
print(f"Provider: {response.metrics.provider_used}")
```

## Project Structure

```
dynamic-prompt-orchestrator/
├── prompt_orchestrator.py          Core prompt engine
├── context_manager.py              State + orchestration layer
├── llm_connector.py                Multi-backend LLM interface
├── main.py                         ETL ingestion entrypoint
├── data/                           Input documents (PDFs, Word, TXT)
├── vector_store/                   FAISS index (output)
├── ingestion/
│   ├── file_loader.py             Document loader
│   ├── chunker.py                 Semantic chunker
│   └── embedder.py                FAISS embedder
├── data_store.json                Persistent state (auto-created)
├── requirements.txt
├── ARCHITECTURE.md                 Detailed architecture guide
└── README.md                       This file
```

## Configuration

### LLM Backends

```python
# Groq (recommended - free, fast)
coordinator = ContextPipelineCoordinator(provider="groq")
# Requires: export GROQ_API_KEY=gsk_...

# Ollama (local - unlimited, free)
coordinator = ContextPipelineCoordinator(provider="ollama")
# Requires: ollama serve (in another terminal)

# Anthropic (cloud - paid)
coordinator = ContextPipelineCoordinator(provider="anthropic")
# Requires: export ANTHROPIC_API_KEY=sk-ant-...

# Mock (testing)
coordinator = ContextPipelineCoordinator(provider="mock")
# Always available, deterministic
```

### Embedding Model

```python
from ingestion.embedder import DocumentEmbedder

# Default: all-MiniLM-L6-v2 (384-dim, fast)
embedder = DocumentEmbedder(
    model_name="all-MiniLM-L6-v2",
    store_dir="vector_store"
)

# Or use larger models for higher quality:
# embedder = DocumentEmbedder(model_name="all-mpnet-base-v2")
# embedder = DocumentEmbedder(model_name="bge-large-en-v1.5")
```

### Chunk Size

```python
# Adjust based on document complexity and LLM context window
chunks = chunk_all(
    documents,
    max_chunk_tokens=512,  # Increase for longer chunks
    preserve_structure=True  # Respect paragraph boundaries
)
```

## Core Components

### Ingestion Pipeline

**file_loader.py** - Loads documents (PDF, Word, TXT, Markdown)
```python
from ingestion.file_loader import DocumentFileLoader

loader = DocumentFileLoader(data_dir="data", recursive=True)
documents = loader.load_files()  # Returns list of {content, file_name, ...}
```

**chunker.py** - Semantic chunking respecting document structure
```python
from ingestion.chunker import chunk_all

chunks = chunk_all(documents, max_chunk_tokens=512)
# Returns list of {chunk_id, content, document_name, token_count, ...}
```

**embedder.py** - Embeddings + FAISS indexing
```python
from ingestion.embedder import DocumentEmbedder

embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
embedder.embed_and_store(chunks)  # Creates vector_store/
results = embedder.search("machine learning", k=5)
```

### Prompt Orchestration Pipeline

**prompt_orchestrator.py** - Dynamic prompt construction
```python
from prompt_orchestrator import (
    DynamicPromptOrchestrator,
    JiraTicket,
    ProductHistory,
    UserContext
)

orchestrator = DynamicPromptOrchestrator()
prompt = orchestrator.generate_prompt(
    ticket=JiraTicket(...),
    history=ProductHistory(...),
    context=UserContext(...)
)
```

**llm_connector.py** - Multi-backend LLM execution
```python
from llm_connector import LLMConnector

connector = LLMConnector(provider="groq")
response = connector.execute(prompt)
# Returns: SegmentedResponse with output + metrics
```

**context_manager.py** - Unified orchestration + state
```python
from context_manager import ContextPipelineCoordinator

coordinator = ContextPipelineCoordinator(provider="groq")
response = coordinator.run_ticket_pipeline(raw_ticket, user)
# Handles: loading state → generating prompt → executing LLM → logging
```

## Advanced Usage: RAG Integration

Combine knowledge base with prompt orchestration:

```python
from ingestion.embedder import DocumentEmbedder
from context_manager import ContextPipelineCoordinator

# Load existing FAISS index
embedder = DocumentEmbedder(store_dir="vector_store")

# Search for relevant documents
query = "How do we implement caching?"
retrieved = embedder.search(query, k=5)

# Augment prompt with retrieved context
context_text = "\n".join([
    f"- {r['document_name']}: {r['content_preview']}"
    for r in retrieved
])

# Create augmented prompt
augmented_prompt = f"""
You are helping with this task:
{original_prompt}

Relevant documentation:
{context_text}

Please provide a comprehensive answer based on the documentation above.
"""

# Execute with LLM
connector = LLMConnector(provider="groq")
response = connector.execute(augmented_prompt)
```

## Testing

Each module includes smoke tests:

```bash
# Test document loading
python ingestion/file_loader.py

# Test semantic chunking
python ingestion/chunker.py

# Test embedding + FAISS
python ingestion/embedder.py

# Test LLM backends
python llm_connector.py

# Test prompt generation
python prompt_orchestrator.py

# Test full orchestration
python context_manager.py
```

## Environment Variables

```bash
# Required for Groq backend
export GROQ_API_KEY=gsk_...

# Optional: Anthropic backend
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: Ollama settings
export OLLAMA_HOST=http://localhost:11434
```

## API Reference

### ContextPipelineCoordinator

```python
coordinator = ContextPipelineCoordinator(
    provider="groq",              # or "ollama", "anthropic", "mock"
    storage_path="data_store.json" # Persistence file
)

response = coordinator.run_ticket_pipeline(
    raw_ticket_data={"id": "...", "summary": "...", ...},
    active_user=UserContext(user_role="...", preferences={...})
)

# Returns SegmentedResponse:
# - final_output: str (LLM response)
# - metrics: ExecutionMetrics (provider, latency, tokens)
# - reasoning_chain: Optional[str]
# - structured_data: Dict[str, Any]
```

### DocumentEmbedder

```python
embedder = DocumentEmbedder(
    model_name="all-MiniLM-L6-v2",
    store_dir="vector_store"
)

# Embed documents
embedder.embed_and_store(chunks)  # Returns summary dict

# Search
results = embedder.search(
    query="your question",
    k=5  # Number of results
)
# Returns list of {chunk_id, document_name, similarity_score, ...}
```

### LLMConnector

```python
connector = LLMConnector(provider="groq")  # or "ollama", "anthropic", "mock"

response = connector.execute(
    prompt="your prompt here",
    system_instructions="optional system context"  # For Anthropic
)

# Returns SegmentedResponse:
# response.final_output: str
# response.metrics.provider_used: str
# response.metrics.execution_time_seconds: float
```

## Performance Metrics

Approximate metrics on typical hardware:

| Operation | Time |
|-----------|------|
| Load 10 PDFs | 2-5s |
| Chunk into 100 chunks | <1s |
| Embed 100 chunks | 30-60s |
| Search FAISS index | <10ms |
| Groq LLM execution | 1-3s |
| Ollama (local, 7B) | 2-5s |

## Cost Estimate

| Provider | Cost (monthly) |
|----------|----------------|
| Groq (free tier) | $0 |
| Ollama (local) | $0 |
| Anthropic (production) | $0.01-1.00 |
| All-MiniLM embeddings | $0 (local) |

## Resume Talking Points

This project demonstrates:

✅ **Enterprise Architecture**
  - Modular design with independent pipelines
  - Multi-provider LLM abstraction
  - Type-safe Pydantic validation
  - Persistent state management

✅ **RAG Implementation**
  - Semantic chunking respecting document structure
  - Dense vector embeddings + FAISS indexing
  - Similarity-based context retrieval
  - RAG prompt augmentation

✅ **LLM Engineering**
  - Few-shot prompting with structured examples
  - Dynamic context injection
  - Multi-provider cost optimization
  - Chain-of-thought reasoning

✅ **Production Patterns**
  - Comprehensive error handling & logging
  - Configurable components
  - Metadata preservation for traceability
  - Built-in smoke tests

## Troubleshooting

### "No module named ingestion"
```bash
# Make sure you're in the project root directory
cd dynamic-prompt-orchestrator

# And ingestion/ directory exists with __init__.py
mkdir -p ingestion/
touch ingestion/__init__.py
```

### "GROQ_API_KEY not found"
```bash
# Set your Groq API key
export GROQ_API_KEY=gsk_...

# Or use mock provider for testing
coordinator = ContextPipelineCoordinator(provider="mock")
```

### "No documents found in data/"
```bash
# Create data directory and add documents
mkdir -p data/
cp /path/to/your/documents.pdf data/
```

### "FAISS import error"
```bash
# Install FAISS CPU version
pip install faiss-cpu

# Or GPU version if available
pip install faiss-gpu
```

## Contributing

This is a portfolio project. Enhancements welcome:

- [ ] Hybrid search (BM25 + dense)
- [ ] Re-ranking with cross-encoders
- [ ] Query expansion strategies
- [ ] FastAPI wrapper for production deployment
- [ ] Streamlit UI
- [ ] Distributed indexing for 10M+ documents

## License

MIT

## Author

Built as a production-grade showcase for RAG + prompt orchestration.

---

**Questions or issues?** Check ARCHITECTURE.md for detailed component documentation.
