"""Text chunker for segmenting parsed book chapters into embedding-ready chunks."""

from dataclasses import dataclass, asdict
from typing import List, Union
from cadence.ingestion.epub_parser import Chapter


@dataclass
class Chunk:
    chapter_index: int
    chapter_title: str
    chunk_index: int
    text: str
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(
    text: str,
    chapter_title: str = "Unknown",
    chapter_index: int = 0,
    max_chars: int = 1500,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Split text into overlapping chunks, preferring paragraph or sentence boundaries.

    Args:
        text: The clean plaintext string.
        chapter_title: Name of the chapter the text belongs to.
        chapter_index: Index of the chapter within the book.
        max_chars: Target maximum character length for each chunk.
        overlap: Number of characters to overlap between successive chunks.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    start = 0
    text_len = len(text)
    chunk_idx = 0

    while start < text_len:
        end = start + max_chars

        if end >= text_len:
            chunk_slice = text[start:].strip()
            if chunk_slice:
                chunks.append(
                    Chunk(
                        chapter_index=chapter_index,
                        chapter_title=chapter_title,
                        chunk_index=chunk_idx,
                        text=chunk_slice,
                        char_count=len(chunk_slice),
                    )
                )
            break

        chunk_slice = text[start:end]

        # Look for natural breaking points within the latter portion of the chunk
        min_break_pos = int(max_chars * 0.5)

        # 1. Paragraph boundary (\n\n)
        last_para = chunk_slice.rfind("\n\n")
        # 2. Sentence boundary (. , ? , ! followed by space or newline)
        last_sentence = max(
            chunk_slice.rfind(". "),
            chunk_slice.rfind(".\n"),
            chunk_slice.rfind("? "),
            chunk_slice.rfind("! "),
        )
        # 3. Line break (\n)
        last_line = chunk_slice.rfind("\n")

        break_at = max(last_para, last_sentence, last_line)

        if break_at > min_break_pos:
            # Found good boundary
            end = start + break_at + 1
            chunk_slice = text[start:end]

        clean_text = chunk_slice.strip()
        if clean_text:
            chunks.append(
                Chunk(
                    chapter_index=chapter_index,
                    chapter_title=chapter_title,
                    chunk_index=chunk_idx,
                    text=clean_text,
                    char_count=len(clean_text),
                )
            )
            chunk_idx += 1

        # Advance start position taking overlap into account
        start = max(start + 1, end - overlap)

    return chunks


def chunk_chapters(
    chapters: List[Chapter],
    max_chars: int = 1500,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Process a list of Chapter objects and split each chapter into chunks.
    
    Args:
        chapters: List of Chapter objects from parse_epub.
        max_chars: Target max characters per chunk.
        overlap: Character overlap between chunks.

    Returns:
        List of all chunks across all chapters.
    """
    all_chunks: List[Chunk] = []
    global_chunk_idx = 0

    for chapter in chapters:
        chapter_chunks = chunk_text(
            text=chapter.content,
            chapter_title=chapter.title,
            chapter_index=chapter.index,
            max_chars=max_chars,
            overlap=overlap,
        )
        for c in chapter_chunks:
            c.chunk_index = global_chunk_idx
            all_chunks.append(c)
            global_chunk_idx += 1

    return all_chunks
