"""Unit and integration tests for Phase 3: ChromaDB vector store, embedder, and RAG agent."""

import pytest
import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

from cadence.memory.vector_store import (
    sanitize_collection_name,
    get_chroma_client,
    get_or_create_collection,
    list_collections,
)
from cadence.memory.embedder import embed_texts, upsert_chunks
from cadence.agents.rag_agent import retrieve_context, answer_question
from cadence.ingestion.chunker import Chunk
from cadence.main import app


@pytest.fixture(scope="module")
def temp_chroma_dir(tmp_path_factory):
    """Temporary directory for ChromaDB test client."""
    temp_dir = tmp_path_factory.mktemp("test_chroma")
    yield str(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_sanitize_collection_name():
    """Verify ChromaDB collection name sanitization conforms to rules."""
    assert sanitize_collection_name("Designing Data-Intensive Applications") == "designing_data-intensive_applications"
    assert sanitize_collection_name("Atomic Habits: An Easy & Proven Way...") == "atomic_habits_an_easy_proven_way"
    assert sanitize_collection_name("!@#$%^&*()") == "book_collection"
    assert len(sanitize_collection_name("a" * 100)) <= 63


def test_chroma_get_or_create_collection(temp_chroma_dir: str):
    """Verify ChromaDB client initializes and creates collections."""
    client = get_chroma_client(persist_dir=temp_chroma_dir)
    collection = get_or_create_collection(client, "Test Book Title")

    assert collection is not None
    assert "test_book_title" in collection.name
    assert collection.name in list_collections(client)


def test_embed_and_retrieve_context(temp_chroma_dir: str):
    """Verify that chunks are embedded, indexed, and retrieved by semantic relevance."""
    client = get_chroma_client(persist_dir=temp_chroma_dir)
    collection = get_or_create_collection(client, "DDIA_Sample")

    sample_chunks = [
        Chunk(
            chapter_index=1,
            chapter_title="Chapter 5. Replication",
            chunk_index=0,
            text="In leaderless replication, clients send each write directly to multiple replicas without a designated master.",
            char_count=107,
        ),
        Chunk(
            chapter_index=2,
            chapter_title="Chapter 7. Transactions",
            chunk_index=1,
            text="ACID transactions provide atomicity, consistency, isolation, and durability guarantees for databases.",
            char_count=104,
        ),
        Chunk(
            chapter_index=3,
            chapter_title="Chapter 3. Storage",
            chunk_index=2,
            text="LSM-trees and B-trees are two dominant indexing storage engines used in modern storage architectures.",
            char_count=106,
        ),
    ]

    # Upsert chunks
    upserted = upsert_chunks(collection, sample_chunks, book_title="DDIA")
    assert upserted == 3

    # Retrieve context for replication
    results = retrieve_context("How does leaderless replication work?", collection=collection, n_results=1)
    assert len(results) == 1
    assert "Replication" in results[0]["chapter_title"]
    assert "leaderless" in results[0]["text"]


def test_answer_question_grounded(temp_chroma_dir: str):
    """Verify that RAG generates an answer grounded in the retrieved context."""
    client = get_chroma_client(persist_dir=temp_chroma_dir)
    collection = get_or_create_collection(client, "DDIA_Sample")

    answer = answer_question("What guarantees do ACID transactions provide?", collection=collection, n_results=2)
    assert len(answer) > 20
    assert any(term in answer.lower() for term in ["atomicity", "isolation", "durability", "acid"])


@pytest.mark.asyncio
async def test_book_chat_intent_routes_to_rag():
    """Verify that asking a question about a book routes to rag_node."""
    config = {"configurable": {"thread_id": "test-rag-routing-1"}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="What does the book say about replication in Chapter 5?")]},
        config=config,
    )

    assert result.get("intent") == "book_chat"
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
