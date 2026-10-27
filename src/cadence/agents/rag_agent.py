"""RAG query pipeline for answering questions grounded in book chunks."""

import asyncio
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from chromadb.api.models.Collection import Collection

from cadence.memory.embedder import embed_texts


def retrieve_context(
    question: str,
    collection: Collection,
    n_results: int = 5,
    embedding_model: str = "gemini-embedding-001",
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most semantically relevant chunks for a question.
    
    Args:
        question: User's query about the book.
        collection: The ChromaDB collection to search within.
        n_results: Number of chunks to retrieve.
        embedding_model: Embedding model name.

    Returns:
        List of dicts containing chunk text, chapter title, and distance score.
    """
    if not question.strip():
        return []

    # 1. Embed query
    query_embeddings = embed_texts([question], model=embedding_model)
    if not query_embeddings:
        return []

    # 2. Query ChromaDB
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(documents)

    retrieved: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc,
            "chapter_title": meta.get("chapter_title", "Unknown"),
            "chapter_index": meta.get("chapter_index", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "book_title": meta.get("book_title", ""),
            "distance": dist,
        })

    return retrieved


def answer_question(
    question: str,
    collection: Collection,
    n_results: int = 5,
    llm_model: str = "gemini-3.5-flash-lite",
    embedding_model: str = "gemini-embedding-001",
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer using Gemini.

    Args:
        question: User's question about the book.
        collection: ChromaDB collection for the target book.
        n_results: Number of context chunks to retrieve.
        llm_model: Generation model name.
        embedding_model: Embedding model name.

    Returns:
        Grounded answer string with chapter citations.
    """
    retrieved_chunks = retrieve_context(
        question=question,
        collection=collection,
        n_results=n_results,
        embedding_model=embedding_model,
    )

    if not retrieved_chunks:
        return "I could not find any relevant sections in the book to answer your question."

    # Format context with citations
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Excerpt {idx} - Chapter: {chunk['chapter_title']}]\n{chunk['text']}"
        )
    formatted_context = "\n\n---\n\n".join(context_blocks)

    system_instruction = (
        "You are Cadence, an insightful AI book companion and reading assistant. "
        "Answer the user's question accurately and thoroughly using ONLY the provided excerpts from the book. "
        "Cite the relevant chapter names when presenting key points. "
        "If the provided context does not contain enough information to answer the question, state that clearly."
    )

    prompt = (
        f"Context from the book:\n{formatted_context}\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, well-structured, and grounded answer based on the excerpts above."
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    return response.text.strip()


async def aanswer_question(
    question: str,
    collection: Collection,
    n_results: int = 5,
    llm_model: str = "gemini-3.5-flash-lite",
    embedding_model: str = "gemini-embedding-001",
) -> str:
    """Async wrapper for answer_question."""
    return await asyncio.to_thread(
        answer_question,
        question=question,
        collection=collection,
        n_results=n_results,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
