import json
import numpy as np
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHUNKS_PATH = Path("rag_chunks.json")
VECTOR_STORE_DIR = Path("vector_store")


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def create_documents(chunks: list[dict]) -> list[Document]:
    docs = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "filename": metadata.get("filename", ""),
                    "source": metadata.get("source", ""),
                    "last_updated": metadata.get("last_updated", ""),
                    "h1": metadata.get("h1", ""),
                    "h2": metadata.get("h2", ""),
                    "h3": metadata.get("h3", ""),
                }
            )
        )
    return docs


def build_vector_store():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Creating documents...")
    docs = create_documents(chunks)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building vector store (this may take a while)...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    vector_store.persist()

    print(f"Vector store saved to {VECTOR_STORE_DIR}")
    return vector_store


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings
    )
    return vector_store


def similarity_search(query: str, k: int = 5, vector_store=None) -> list[dict]:
    if vector_store is None:
        vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "filename": doc.metadata.get("filename", ""),
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content,
            "score": float(score)
        })

    return formatted_results


if __name__ == "__main__":
    vector_store = build_vector_store()

    print("\n--- Testing similarity search ---")
    results = similarity_search("What is the syllabus for the AI course?", k=3, vector_store=vector_store)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"File: {r['filename']}")
        print(f"Content preview: {r['content'][:300]}...")
