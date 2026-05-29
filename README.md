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

- `GOOGLE_API_KEY` - Get from [Google AI Studio](https://aistudio.google.com/app/apikey)

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

### Hour 5: Run ADK Agent

```bash
python agent.py
```

Starts an interactive agent that can:
- Answer questions about your notes (`ask_question`)
- Search for relevant chunks (`search_notes`)
- List available notes (`list_available_notes`)
- Use web search as fallback (`google_search`)
