# Obsidian RAG with ADK Agent

RAG pipeline for querying your Obsidian notes using Google ADK agent.

## Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r req.txt
```

## Environment Setup

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

- `GROQ_API_KEY` - Get from [Groq Cloud](https://console.groq.com/keys)

## Pipeline Steps

### Hour 1-2: Export & Chunk (already done)

```bash
python chunk.py
```

Creates `rag_chunks.json` from your Obsidian export.

### Hour 3: Build Vector Store

```bash
python embeddings.py
```

Creates `vector_store/` with Chroma embeddings.

### Hour 4: Test RAG (optional)

```bash
python llm_answer.py
```

CLI to test RAG answers directly.

### Project Structure

```
obsidian/          # ADK agent directory
├── __init__.py
├── agent.py       # Agent definition with tools
└── .env           # API key config
```

### Hour 5: Run ADK Agent

```bash
adk run obsidian
# or with a single query:
adk run obsidian "your question"
```

Starts an interactive agent (powered by `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` via Groq) that can:
- Answer questions about your notes (`ask_question`)
- Search for relevant chunks (`search_notes`)
- List available notes (`list_available_notes`)
