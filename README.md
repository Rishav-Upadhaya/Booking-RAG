# Interview-Booking RAG

Document RAG backend: ingestion, hybrid retrieval, and conversational RAG with tool-calling agents for interview booking.

## Features

- **Upload & ingest** PDF/TXT documents, chunk them (`fixed` token window or `recursive` structure-aware), embed, and index in Pinecone, all in the background
- **Hybrid retrieval**: dense vectors + `pinecone-sparse-english-v0` BM25-style sparse search, re-ranked with `bge-reranker-v2-m3`
- **Conversational RAG**: LangChain `create_agent` (LangGraph runtime) with retrieval, PII redaction, and conversation summarization middleware
- **Intent classification**: rule + embedding-based routing (domain questions, booking, greeting, guardrail, recall, off-topic)
- **Interview booking**: deterministic booking flow (extract → normalize → validate → persist), LLM only in free-text normalization
- **Session memory**: Redis-backed, per-session, TTL-expiring history with edit/retry support
- **Streaming**: Server-Sent Events for chat replies with incremental tokens, sources, and token usage

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│  FastAPI    │──▶│  Ingestion   │──▶│    Vector    │   │   Postgres  │
│  (routes)   │   │ extract →    │   │    store     │   │ documents / │
│             │   │ chunk →      │   │  (Pinecone)  │   │ chunks /    │
│             │   │ embed        │   │              │   │ bookings    │
└─────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
       │                                                            ▲
       │        ┌───────────────────────────────────────────────────┘
       ▼        │
   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
   │  Chat       │──▶│  Intent      │──▶│     RAG      │──▶│    Redis    │
   │  (SSE)      │   │  classify    │   │  agent loop  │   │  memory     │
   │             │   │  booking     │   │  middlewares │   │  partials   │
   └─────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
```

- `app/api/`: FastAPI routes
- `app/chunking/`, `app/extraction/`, `app/embeddings/`, `app/vectorstore/`: ingestion pipeline
- `app/rag/`, `app/intent/`, `app/retrieval/`: chat/agent pipeline
- `app/booking/`, `app/services/`: booking and persistence services
- `app/memory/`: Redis session memory
- `alembic/`: schema migrations

## Tech Stack

| Component    | Choice                                                        |
| ------------ | ------------------------------------------------------------- |
| API          | FastAPI + Uvicorn                                             |
| Runtime      | LangChain `create_agent` on LangGraph                         |
| Vector store | Pinecone (serverless): dense + sparse hybrid, hosted rerank  |
| Embeddings   | Jina `jina-embeddings-v3` (cloud) or `all-MiniLM-L6-v2` (local) |
| LLM          | OpenRouter (OpenAI-compatible endpoint)                       |
| Memory       | Redis                                                         |
| Storage      | PostgreSQL via SQLAlchemy 2.0 + Alembic                       |

## Getting Started

### Prerequisites

- Docker (Postgres 16, Redis 7)
- Python 3.11+
- A Pinecone API key (vector store + hosted sparse/rerank models)
- An OpenRouter API key (LLM)

### Run with Docker

```bash
cp .env.example .env
# fill in PINECONE_API_KEY, OPENROUTER_API_KEY (JINA_API_KEY optional)
docker compose up -d --build
```

The app container runs Alembic migrations and a startup self-test on every boot:

- resolves the active embedder and verifies it can embed a probe string
- verifies the Pinecone index dimension matches the active embedder (aborts with a clear message if you switch embedding providers without recreating the index)

Interactive API docs: <http://localhost:8001/docs>

### Run locally

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
docker compose up -d postgres redis
alembic upgrade head
.venv/bin/uvicorn app.main:app --port 8001
```

### Tests

```bash
.venv/bin/pytest
```

Tests requiring Redis or the local embedding model skip automatically when unavailable.

## Configuration

All settings are read from `.env` (see `.env.example` for the full list).

### Embedding provider

`EMBEDDER_PROVIDER`:

| Value    | Behavior                                                                  |
| -------- | ------------------------------------------------------------------------- |
| `auto`   | Jina cloud embeddings when `JINA_API_KEY` is set, otherwise local model   |
| `local`  | Always use the local `sentence-transformers` model (downloaded on boot)   |
| `jina`   | Require `JINA_API_KEY` (fails fast at startup if missing)                 |

> Vectors are model-specific: switching providers without recreating the Pinecone index will abort at startup.

### Retrieval

| Variable                 | Default | Description                                  |
| ------------------------ | ------- | -------------------------------------------- |
| `TOP_K`                  | `5`     | Final results after reranking                |
| `RETRIEVAL_CANDIDATES`   | `20`    | Candidates fetched from Pinecone before rerank |

### Ingestion

| Variable                  | Default     | Description                       |
| ------------------------- | ----------- | --------------------------------- |
| `MAX_UPLOAD_BYTES`        | `52428800`  | 50 MB upload limit                |
| `EMBED_BATCH_SIZE`        | `64`        | Embedding batch size              |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Token window and overlap for chunking |
| `INTENT_DOMAIN_THRESHOLD` | `0.15`      | Cosine cutoff before falling back to `domain` intent |

## API

| Method | Path                          | Description                                          |
| ------ | ----------------------------- | ---------------------------------------------------- |
| POST   | `/documents/upload`           | Upload PDF/TXT; `chunking_strategy=fixed\|recursive` (default `fixed`); processing runs in background |
| GET    | `/documents`                  | List documents and their status/chunk counts         |
| GET    | `/documents/{id}/status`      | Status, chunk count, error                           |
| GET    | `/documents/{id}/chunks`      | Chunks of a document                                 |
| DELETE | `/documents/{id}`             | Delete document, its chunks, and its vectors         |
| POST   | `/chat/message`               | Ask a question (RAG answer + sources + token usage)  |
| POST   | `/chat/stream`                | Ask a question via SSE (sources → deltas → done)     |
| GET    | `/chat/sessions`              | List conversation sessions                           |
| GET    | `/chat/{session_id}/history`  | Conversation history for a session                   |
| POST   | `/chat/{session_id}/rewind`   | Truncate history to `keep` turns (edit/retry)        |
| DELETE | `/chat/{session_id}`          | Delete a session                                     |
| POST   | `/bookings`                   | Create a booking directly                            |
| GET    | `/bookings/{id}`              | Fetch a booking                                      |
| GET    | `/health`                     | Liveness (checks Postgres + Redis)                   |

## Data Model

| Table       | Contents                                              |
| ----------- | ----------------------------------------------------- |
| `documents` | filename, content type, status, chunking strategy, chunk count, error |
| `chunks`    | text, source page, vector id: one row per indexed vector |
| `bookings`  | name, email, date, time, session id                   |

`documents.status` transitions: `pending` → `processing` → `ready` / `failed` (with `error` message).

## Security & Privacy

- URLs, credit-card numbers, and IP addresses are redacted from conversations before they reach the model (PII middleware)
- Email addresses are intentionally kept because the booking tool captures them
- Prompt-injection/guardrail patterns are intercepted before the LLM
