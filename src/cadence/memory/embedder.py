"""Batch embedding generation via Google Gemini API and ChromaDB upsert."""

import asyncio
from typing import List, Union, Dict, Any
from google import genai
from chromadb.api.models.Collection import Collection

from cadence.ingestion.chunker import Chunk


def embed_texts(
    texts: List[str],
    model: str = "gemini-embedding-001",
    batch_size: int = 50,
) -> List[List[float]]:
    """
    Generate vector embeddings for a list of strings using Google Gemini in batches.
    
    Args:
        texts: List of text snippets to embed.
        model: Google GenAI embedding model name.
        batch_size: Maximum items per batch request.

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    if not texts:
        return []

    client = genai.Client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=batch,
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

    return all_embeddings


def upsert_chunks(
    collection: Collection,
    chunks: List[Union[Chunk, Dict[str, Any]]],
    book_title: str = "unknown_book",
    model: str = "gemini-embedding-001",
    batch_size: int = 50,
) -> int:
    """
    Embed and upsert a list of chunks into a ChromaDB collection.

    Args:
        collection: The ChromaDB collection instance.
        chunks: List of Chunk objects or dicts.
        book_title: Title of the book for metadata tagging.
        model: Embedding model name.
        batch_size: Batch size for embeddings and upserts.

    Returns:
        Total number of chunks successfully embedded and upserted.
    """
    if not chunks:
        return 0

    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # Extract text, ids, and metadata from Chunk objects or dicts
        texts = []
        ids = []
        metadatas = []

        for c in batch:
            if isinstance(c, Chunk):
                text = c.text
                chunk_idx = c.chunk_index
                ch_title = c.chapter_title
                ch_idx = c.chapter_index
                chars = c.char_count
            else:
                text = c["text"]
                chunk_idx = c.get("chunk_index", 0)
                ch_title = c.get("chapter_title", "Unknown")
                ch_idx = c.get("chapter_index", 0)
                chars = c.get("char_count", len(text))

            texts.append(text)
            ids.append(f"chunk_{chunk_idx}")
            metadatas.append({
                "book_title": book_title,
                "chapter_index": ch_idx,
                "chapter_title": ch_title,
                "chunk_index": chunk_idx,
                "char_count": chars,
            })

        # Generate embeddings for the batch
        embeddings = embed_texts(texts, model=model, batch_size=batch_size)

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_upserted += len(batch)

    return total_upserted


async def aupsert_chunks(
    collection: Collection,
    chunks: List[Union[Chunk, Dict[str, Any]]],
    book_title: str = "unknown_book",
    model: str = "gemini-embedding-001",
    batch_size: int = 50,
) -> int:
    """Async wrapper to embed and upsert chunks without blocking the event loop."""
    return await asyncio.to_thread(
        upsert_chunks,
        collection=collection,
        chunks=chunks,
        book_title=book_title,
        model=model,
        batch_size=batch_size,
    )
