import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from embeddings import similarity_search, load_vector_store

MAX_CHUNKS = 5
DEFAULT_K = 5

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful AI assistant answering questions about the user's Obsidian notes.

When answering:
1. Use only the provided context chunks from the user's notes
2. Cite sources by filename when possible
3. If the context doesn't contain enough information, say so clearly
4. Keep answers concise but informative
5. Format code snippets if present in the context

Context from user's notes:"""

USER_PROMPT_TEMPLATE = """Question: {question}

Relevant chunks from the knowledge base:
{context}

Please provide a helpful answer based on the context above."""


def format_context(results: list[dict]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        filename = r.get("filename", "Unknown")
        content = r.get("content", "")
        context_parts.append(f"--- Chunk {i} (from {filename}) ---\n{content}")
    return "\n\n".join(context_parts)


def get_answer(query: str, k: int = DEFAULT_K) -> dict:
    vector_store = load_vector_store()
    results = similarity_search(query, k=k, vector_store=vector_store)

    context = format_context(results)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=query, context=context)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [{"filename": r["filename"], "chunk_id": r["chunk_id"]} for r in results],
        "num_chunks_used": len(results)
    }


if __name__ == "__main__":
    print("=== RAG Question Answering ===")
    print("Ask questions about your Obsidian notes (or 'quit' to exit)\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            continue

        print("\nSearching and generating answer...")
        result = get_answer(query)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(s['filename'] for s in result['sources'])}")
        print(f"Chunks used: {result['num_chunks_used']}\n")
