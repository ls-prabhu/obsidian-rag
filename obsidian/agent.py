import json
import os
import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from embeddings import load_vector_store, similarity_search

# Preload vector store at module level so the embedding model is loaded once at startup
_vector_store = load_vector_store()
AGENT_NAME = "obsidian_rag_agent"
AGENT_DESCRIPTION = "Answers questions about your Obsidian notes using RAG"


def search_notes(query: str, k: int = 5) -> dict:
    """Search your Obsidian notes for relevant content."""
    vector_store = load_vector_store()
    results = similarity_search(query, k=k, vector_store=vector_store)

    if not results:
        return {
            "status": "no_results",
            "message": "No relevant chunks found for your query.",
            "results": []
        }

    formatted = []
    for r in results:
        formatted.append({
            "filename": r["filename"],
            "content_preview": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
            "score": r["score"]
        })

    return {
        "status": "success",
        "query": query,
        "num_results": len(formatted),
        "results": formatted
    }


def ask_question(query: str) -> dict:
    """Get a full RAG answer from your Obsidian notes. Use this for most questions."""
    from langchain_core.messages import HumanMessage, SystemMessage

    vector_store = load_vector_store()
    results = similarity_search(query, k=5, vector_store=vector_store)

    if not results:
        return {
            "status": "no_results",
            "answer": "No relevant information found in your notes for this query.",
            "sources": []
        }

    context_parts = []
    for i, r in enumerate(results, 1):
        filename = r.get("filename", "Unknown")
        content = r.get("content", "")
        context_parts.append(f"--- Chunk {i} (from {filename}) ---\n{content}")

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful AI assistant answering questions about the user's Obsidian notes.

When answering:
1. Use only the provided context chunks from the user's notes
2. Cite sources by filename when possible
3. If the context doesn't contain enough information, say so clearly
4. Keep answers concise but informative"""

    user_prompt = f"""Question: {query}

Relevant chunks from your notes:
{context}

Provide a helpful answer based on the context above."""

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)

        return {
            "status": "success",
            "answer": response.content,
            "sources": [{"filename": r["filename"], "chunk_id": r["chunk_id"]} for r in results],
            "num_chunks_used": len(results)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "answer": "Sorry, I encountered an error while processing your question.",
            "sources": []
        }


def list_available_notes() -> dict:
    """List all indexed Obsidian notes."""
    chunks_path = ROOT_DIR / "rag_chunks.json"
    if not chunks_path.exists():
        return {
            "status": "error",
            "message": "No chunks found. Run chunk.py first."
        }

    with chunks_path.open(encoding="utf-8") as f:
        chunks = json.load(f)

    filenames = set(c["metadata"]["filename"] for c in chunks)
    return {
        "status": "success",
        "total_notes": len(filenames),
        "notes": sorted(filenames)
    }


root_agent = Agent(
    name=AGENT_NAME,
    model="groq/llama-3.1-8b-instant",
    description=AGENT_DESCRIPTION,
    tools=[
        FunctionTool(search_notes),
        FunctionTool(ask_question),
        FunctionTool(list_available_notes),
    ],
    instruction="""You are a helpful assistant that helps the user answer questions about their Obsidian notes.

You have access to these tools:
1. `ask_question` - Get a full RAG answer from the notes (recommended for most queries)
2. `search_notes` - Search for relevant chunks without generating an answer  
3. `list_available_notes` - See what notes are indexed

For user questions about their notes, use `ask_question`. Use `search_notes` if you want to see raw chunks first. Use `list_available_notes` if the user wants to know what topics are available."""
)
