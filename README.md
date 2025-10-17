# RAG System Backend

Role-based RAG system with LangChain and ChromaDB for the BadCompany game.

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-dotenv langchain langchain-openai langchain-community chromadb redis sentence-transformers docx2txt
```

### 2. Configure Environment

Create `.env` file:
```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL_NAME=meta-llama/llama-3.3-70b-instruct
```

### 3. Run Server

```bash
uvicorn server:app --reload --port 8000 --host 0.0.0.0
```

Server will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /agent/chat` - Chat with RAG system
- `POST /judge/evaluate` - Evaluate attack attempts

## Project Structure

- `server.py` - FastAPI server with RAG endpoints
- `core/` - RAG pipeline (embeddings, retrieval, LLM)
- `data/` - Documents organized by access level (public, worker, admin)
- `users/` - User schema and access control
- `config/` - Settings and configuration
- `vector_store_*/` - ChromaDB vector stores per role

## Role-Based Access

- **public**: Access to public documents only
- **worker**: Access to public + worker documents  
- **admin**: Access to all documents

Documents in `data/admin/` contain sensitive information that attackers try to extract.

