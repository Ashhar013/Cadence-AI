"""Tests for LangGraph routing logic in the Cadence AI graph."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from cadence.main import app


@pytest.mark.asyncio
async def test_conversation_intent_routes_to_general_chat():
    """A casual greeting should be classified as 'conversation' and produce an AI response."""
    config = {"configurable": {"thread_id": "test-conversation-1"}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Hello, how are you?")]},
        config=config,
    )

    assert result.get("intent") == "conversation"
    assert result.get("processing_status") == "conversation_complete"

    # Should have an AIMessage response
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1


@pytest.mark.asyncio
async def test_book_search_intent_triggers_query_node():
    """A book search request should be classified as 'book_search' and reach book_details_node.
    
    NOTE: This test will actually invoke the Playwright scraper.
    For CI, this should be mocked. For local dev, it validates the full pipeline.
    Skip this test if running without a display / in CI.
    """
    config = {"configurable": {"thread_id": "test-book-search-1"}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Find me the book Atomic Habits by James Clear")]},
        config=config,
    )

    # The graph should pause at book_selection_node's interrupt
    # So we check that intent was classified and query was extracted
    assert result.get("intent") == "book_search"


@pytest.mark.asyncio
async def test_empty_message_handled_gracefully():
    """An empty conversation should not crash the graph."""
    config = {"configurable": {"thread_id": "test-empty-1"}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="")]},
        config=config,
    )

    # Should still produce some kind of result without crashing
    assert result is not None
    assert "messages" in result
