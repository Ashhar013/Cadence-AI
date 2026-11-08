
"""Evaluate DDIA retrieval quality using freshly parsed EPUB content."""

import asyncio
import uuid
from pathlib import Path

import pytest

from cadence.ingestion.epub_parser import parse_epub
from cadence.ingestion.chunker import chunk_text
from cadence.memory.vector_store import get_chroma_client
from cadence.memory.embedder import aupsert_chunks
from cadence.agents.rag_agent import retrieve_context


BOOK_TITLE = "Designing Data-Intensive Applications"

EPUB_PATH = Path(
    "data/raw/Designing Data-Intensive Applications_ "
    "The Big Ideas Behind -- Unknown -- 2016 -- "
    "5e70d11b8f8f236bd595024cb43d8234 -- Anna’s Archive.epub"
)

QUESTIONS = [
    "What are the differences between OLAP and OLTP?",
    "Why is column-oriented storage useful for analytical workloads?",
    "How does leaderless replication work?",
]


def test_retrieval_quality():
    # 1. Parse the EPUB using the updated parser.
    assert EPUB_PATH.exists(), f"EPUB not found: {EPUB_PATH}"

    chapters = parse_epub(EPUB_PATH)

    print(f"\nParsed chapters: {len(chapters)}")

    # 2. Generate fresh chunks from the parsed chapters.
    chunks = [
        chunk
        for chapter in chapters
        for chunk in chunk_text(
            text=chapter.content,
            chapter_title=chapter.title,
            chapter_index=chapter.index,
        )
    ]

    assert chunks, "No chunks generated from the EPUB."

    print(f"Fresh chunks generated: {len(chunks)}")

    # 3. Create a unique test-only Chroma collection.
    client = get_chroma_client()
    collection_name = f"test_ddia_{uuid.uuid4().hex[:12]}"

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        # 4. Index the fresh chunks directly.
        indexed_count = asyncio.run(
            aupsert_chunks(
                collection=collection,
                chunks=chunks,
                book_title=BOOK_TITLE,
            )
        )

        print(f"Indexed chunks: {indexed_count}")
        print(f"Test collection: {collection.name}")

        # 5. Evaluate retrieval independently of answer generation.
        for question in QUESTIONS:
            print("\n" + "=" * 90)
            print(f"QUERY: {question}")
            print("=" * 90)

            results = retrieve_context(
                question=question,
                collection=collection,
                n_results=5,
            )

            if not results:
                print("No results returned.")
                continue

            for rank, result in enumerate(results, start=1):
                print(f"\n--- Rank {rank} ---")
                print(f"Chapter: {result['chapter_title']}")
                print(f"Chapter index: {result['chapter_index']}")
                print(f"Chunk index: {result['chunk_index']}")
                print(f"Cosine distance: {result['distance']:.4f}")

                text = result["text"]
                preview = text[:500] + (
                    "..." if len(text) > 500 else ""
                )
                print(f"Text preview:\n{preview}")

    finally:
        # Delete only this test's temporary collection.
        client.delete_collection(name=collection_name)
