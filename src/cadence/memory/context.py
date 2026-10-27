"""Context management and conversation history summarization."""

from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from google import genai
from google.genai import types


def should_summarize(messages: List[BaseMessage], threshold: int = 10) -> bool:
    """Return True if message history exceeds the threshold for summarization."""
    # Count user and assistant turns
    dialog_turns = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    return len(dialog_turns) >= threshold


def split_messages_for_summary(
    messages: List[BaseMessage],
    keep_recent: int = 4,
) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    """
    Split messages into older messages to be summarized and recent messages to keep verbatim.
    """
    if len(messages) <= keep_recent:
        return [], messages
    return messages[:-keep_recent], messages[-keep_recent:]


def summarize_conversation(
    older_messages: List[BaseMessage],
    existing_summary: str = "",
    model: str = "gemini-3.5-flash-lite",
) -> str:
    """
    Compress older conversation turns into a running summary.

    Args:
        older_messages: Older messages to condense.
        existing_summary: Prior summary, if any.
        model: Google GenAI model name.

    Returns:
        Concise summary string capturing key topics, questions, and book titles discussed.
    """
    if not older_messages:
        return existing_summary

    dialog_lines = []
    for m in older_messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else "System"
        content = m.content
        if isinstance(content, list):
            content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
        dialog_lines.append(f"{role}: {str(content)[:300]}")

    formatted_dialog = "\n".join(dialog_lines)

    prompt = (
        "Condense the following conversation history into a concise 2-4 sentence summary. "
        "Preserve key facts, book titles mentioned, specific questions asked, and user preferences.\n\n"
    )
    if existing_summary:
        prompt += f"Existing Summary:\n{existing_summary}\n\n"

    prompt += f"New Conversation to Incorporate:\n{formatted_dialog}\n\nUpdated Summary:"

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )
    return response.text.strip()
