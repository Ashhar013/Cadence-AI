"""Batch embedding generation via Google Gemini API and ChromaDB upsert."""

import asyncio
from typing import List, Union, Dict, Any
from google import genai
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from cadence.ingestion.chunker import Chunk


embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_texts(
    texts: List[str],
    batch_size: int = 50,
) -> List[List[float]]:
    """
    Generate local embeddings using BGE-small.
    """
    if not texts:
        return []

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def upsert_chunks(
    collection: Collection,
    chunks: List[Union[Chunk, Dict[str, Any]]],
    book_title: str = "unknown_book",
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
            ids.append(f"chapter_{ch_idx}_chunk_{chunk_idx}")
            metadatas.append({
                "book_title": book_title,
                "chapter_index": ch_idx,
                "chapter_title": ch_title,
                "chunk_index": chunk_idx,
                "char_count": chars,
            })

        # Generate embeddings for the batch
        embeddings = embed_texts(texts, batch_size=batch_size)

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
    batch_size: int = 50,
) -> int:
    """Async wrapper to embed and upsert chunks without blocking the event loop."""
    return await asyncio.to_thread(
        upsert_chunks,
        collection=collection,
        chunks=chunks,
        book_title=book_title,
        batch_size=batch_size,
    )
