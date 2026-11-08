from sentence_transformers import SentenceTransformer

from cadence.main import find_local_epub
from cadence.ingestion.epub_parser import parse_epub
from cadence.ingestion.chunker import chunk_chapters  # Verify actual module path

# Find the EPUB using your existing helper
epub_path = find_local_epub("Designing Data-Intensive Applications")

if epub_path is None:
    raise FileNotFoundError("Book not found in data/raw.")

print(f"Found EPUB: {epub_path}")

# Parse and chunk
chapters = parse_epub(epub_path)
chunks = chunk_chapters(chapters)

# Load local embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
tokenizer = model.tokenizer

# Measure token counts
token_counts = [
    len(tokenizer.encode(chunk.text, add_special_tokens=True))
    for chunk in chunks
]

print("\n--- Chunk Analysis ---")
print("Total chapters:", len(chapters))
print("Total chunks:", len(chunks))
print("Maximum tokens:", max(token_counts))
print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))
print("Chunks exceeding 512:", sum(t > 512 for t in token_counts))