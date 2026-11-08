"""ePub document parser for extracting chapter-segmented text."""

import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Union

import ebooklib
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from ebooklib import epub

# Suppress harmless ebooklib and bs4 parser warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.getLogger("ebooklib").setLevel(logging.ERROR)


@dataclass
class Chapter:
    index: int
    title: str
    content: str  # Clean plaintext
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _build_toc_map(toc_items: list) -> Dict[str, str]:
    """Recursively build a mapping from ePub item filename to TOC title."""
    mapping: Dict[str, str] = {}

    def _traverse(items):
        for item in items:
            if isinstance(item, tuple):
                node, children = item[0], item[1]
                if hasattr(node, "href") and hasattr(node, "title"):
                    filename = node.href.split("#")[0]
                    if filename not in mapping:
                        mapping[filename] = node.title
                _traverse(children)
            elif hasattr(item, "href") and hasattr(item, "title"):
                filename = item.href.split("#")[0]
                if filename not in mapping:
                    mapping[filename] = item.title

    _traverse(toc_items)
    return mapping


def _remove_embedded_toc(soup: BeautifulSoup) -> None:
    """
    Remove a printed Table of Contents block from an EPUB document.
    """
    if soup.body is None:
        return

    body_elements = soup.body.find_all(recursive=False)

    # Find the exact TOC heading.
    start_index = next(
        (
            i for i, element in enumerate(body_elements)
            if element.get_text(" ", strip=True).casefold()
            == "table of contents"
        ),
        None,
    )

    if start_index is None:
        return

    # Identify actual TOC page footer patterns.
    end_indices = []

    for i in range(start_index + 1, len(body_elements)):
        text = " ".join(
            body_elements[i].get_text(" ", strip=True).casefold().split()
        )

        if (
            text.startswith("table of contents |")
            or text.endswith("| table of contents")
        ):
            end_indices.append(i)

    # Fail safely if no footer is found.
    if not end_indices:
        return

    end_index = end_indices[-1]

    # Remove the complete TOC block.
    for element in body_elements[start_index:end_index + 1]:
        element.decompose()

def parse_epub(file_path: Union[str, Path], min_char_threshold: int = 200) -> List[Chapter]:
    """
    Extract chapters as clean text from an ePub file.
    
    Args:
        file_path: Path to the .epub file.
        min_char_threshold: Minimum characters required to consider an item a valid chapter.
        
    Returns:
        List of Chapter dataclasses containing clean chapter titles and contents.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ePub file not found: {path}")

    # Read ePub archive
    book = epub.read_epub(str(path))
    toc_map = _build_toc_map(book.toc)
    chapters: List[Chapter] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        raw_content = item.get_content()
        if not raw_content:
            continue

        soup = BeautifulSoup(raw_content, "lxml")

        # Strip scripts, styles, and navigational elements
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        # Clean text
        _remove_embedded_toc(soup)
        text = soup.get_text(separator="\n", strip=True)

        # Filter out tiny sections (cover-only, copyright blurb, blank pages)
        if len(text) < min_char_threshold:
            continue

        # 1. First priority: Title from ePub Table of Contents
        item_filename = item.file_name.split("/")[-1]
        title = toc_map.get(item_filename) or toc_map.get(item.file_name)

        # 2. Second priority: Clean heading if TOC didn't match
        if not title:
            for heading_tag in soup.find_all(["h1", "h2", "h3"]):
                candidate_text = heading_tag.get_text(strip=True)
                # Ignore numbers-only or image filenames as titles
                if len(candidate_text) >= 3 and not candidate_text.replace(".", "").isdigit():
                    title = candidate_text
                    break

        # 3. Third priority: Document <title>
        if not title and soup.title and soup.title.get_text(strip=True):
            candidate_text = soup.title.get_text(strip=True)
            if len(candidate_text) >= 3:
                title = candidate_text

        # 4. Fallback
        if not title:
            title = f"Section {len(chapters) + 1}"

        # Skip EPUB-generated document outlines.
        if title.strip().casefold() == "document outline":
            continue

        chapter = Chapter(
            index=len(chapters),
            title=title.strip(),
            content=text,
            char_count=len(text),
        )
        chapters.append(chapter)

    return chapters
