"""RAG query pipeline for answering questions grounded in book chunks."""

import time
from google.genai import errors
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
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most semantically relevant chunks using local BGE embeddings.
    """
    if not question.strip():
        return []

    # 1. Embed query locally using BGE
    query_embeddings = embed_texts([question])

    if not query_embeddings:
        return []

    # 2. Query ChromaDB
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = (
        results.get("distances", [[]])[0]
        if "distances" in results
        else [0.0] * len(documents)
    )

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


def rewrite_retrieval_query(
    question: str,
    chat_history: List[Dict[str, str]],
    llm_model: str = "gemini-3.5-flash-lite",
) -> str:
    """Rewrite a follow-up question into a standalone book retrieval query."""

    if not question.strip() or not chat_history:
        return question

    history = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in chat_history
        if turn.get("content", "").strip()
    )

    system_instruction = (
        "You are a query contextualization assistant for a book RAG system. "
        "Rewrite the latest user question into a standalone question that can "
        "be understood without the conversation history. "
        "Use conversation history only to resolve references and missing context. "
        "Preserve the user's original intent and meaning. "
        "Do not answer the question. "
        "If the question is already standalone, return it unchanged. "
        "Return only the rewritten question, without explanations or quotation marks."
    )

    prompt = (
        f"Conversation history:\n{history}\n\n"
        f"Latest user question:\n{question}"
    )

    client = genai.Client()

    response = client.models.generate_content(
        model=llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )

    rewritten_query = (response.text or "").strip()

    return rewritten_query if rewritten_query else question


def answer_question(
    question: str,
    collection: Collection,
    n_results: int = 5,
    llm_model: str = "gemini-3.5-flash-lite",
    retrieval_query: Optional[str] = None,
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer using Gemini.

    Args:
        question: User's question about the book.
        collection: ChromaDB collection for the target book.
        n_results: Number of context chunks to retrieve.
        llm_model: Generation model name.
        retrieval_query: Optional query to use for retrieving context.
        embedding_model: Embedding model name.

    Returns:
        Grounded answer string with chapter citations.
    """
    retrieved_chunks = retrieve_context(
        question=retrieval_query or question,
        collection=collection,
        n_results=n_results,
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
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return response.text.strip()

        except errors.APIError as e:
            if e.code not in (500, 502, 503, 504):
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(
                f"[WARNING] Gemini returned {e.code}. "
                f"Retrying in {wait_time}s "
                f"({attempt + 1}/{max_retries})"
            )

            time.sleep(wait_time)


async def aanswer_question(
    question: str,
    collection: Collection,
    n_results: int = 5,
    llm_model: str = "gemini-3.5-flash-lite",
    retrieval_query: Optional[str] = None,
) -> str:
    """Async wrapper for answer_question."""
    return await asyncio.to_thread(
        answer_question,
        question=question,
        collection=collection,
        n_results=n_results,
        llm_model=llm_model,
        retrieval_query=retrieval_query,
    )
