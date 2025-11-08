# RAG System Backend

Role-based RAG system with LangChain and ChromaDB for the BadCompany game.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy and configure `.env` file:
```bash
copy .env.example .env
```

Template for `.env` file:
```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=your_project_url
TAVILY_API_KEY=''
HF_TOKEN=your_api_key
USE_HF_EMBEDDINGS=true
HF_LOGGING=true
HF_API_URL=https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5
USER_AGENT=RAG-System/1.0
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed configuration instructions.

### 3. Run Server

```bash
uvicorn server:app --reload --port 8000
```

### 4. Verify Setup

```bash
curl http://127.0.0.1:8000/debug/status
```

Server will be available at `http://localhost:8000`

## Features

- HuggingFace Embeddings Integration
  - Automatic endpoint discovery and fallback
  - Batch processing for efficiency
  - Support for multiple embedding models
  - Fallback to local sentence-transformers

- Role-Based Access Control
  - Public, worker, and admin document access
  - Separate vector stores per role
  - Prevents privilege escalation

- Robust Error Handling
  - Clear error messages for missing credentials
  - Automatic retry with endpoint variations
  - Graceful degradation

## API Endpoints

- `GET /` - Health check
- `GET /debug/status` - Diagnostic information (embeddings, docs, config)
- `POST /session/start` - Initialize session
- `POST /agent/chat` - Chat with RAG system
- `POST /judge/evaluate` - Evaluate attack attempts
- `GET /scenarios` - List available scenarios

## Project Structure

- `server.py` - FastAPI server with RAG endpoints
- `core/` - RAG pipeline (embeddings, retrieval, LLM, vectorstore)
  - `embeddings.py` - HuggingFace embeddings wrapper with auto-discovery
  - `retrieval.py` - Document loading and RAG chain
  - `vectorstore.py` - ChromaDB integration
- `data/` - Documents organized by access level (public, worker, admin)
- `config/` - Settings and credentials
- `vector_store_*/` - ChromaDB vector stores per role

## Role-Based Access

- **public**: Access to public documents only
- **worker**: Access to public + worker documents  
- **admin**: Access to all documents

Documents in `data/admin/` contain sensitive information that attackers try to extract.

## HuggingFace Embeddings

The system uses HuggingFace Inference API by default with intelligent endpoint discovery:

1. Automatic Format Detection: Tries array and string payload formats
2. Batch Processing: Embeds multiple texts in single API call when possible
3. Smart Fallbacks: Tests multiple endpoint variations automatically
4. Local Fallback: Can use sentence-transformers if HF unavailable

Supported models:
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `BAAI/bge-small-en-v1.5` (better quality)
- `BAAI/bge-base-en-v1.5` (best quality)

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for configuration options.

## Troubleshooting

**No embeddings / token errors:**
- Ensure `HF_TOKEN` is set in `.env`
- Get token from https://huggingface.co/settings/tokens

**Slow first request:**
- HF models have 30-60s cold start on first request
- Subsequent requests are fast (model stays loaded)

**Document loading issues:**
- Check `/debug/status` endpoint for diagnostics
- Verify files exist in `data/` subdirectories

**Full troubleshooting guide:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn server:app --reload --port 8000

# Check logs for HF embedding calls
# (Set HF_LOGGING=true in .env)
```

