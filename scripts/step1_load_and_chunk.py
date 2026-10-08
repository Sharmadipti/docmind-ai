"""
DocMind AI - Step 1: Load a PDF and split it into chunks.

Why chunk?
  Embeddings work best on small, focused pieces of text, and later we only
  want to send the relevant pieces to the LLM (not the whole PDF).

Run:
  python step1_load_and_chunk.py path/to/file.pdf
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Starting values from my research notes. To be tuned later with test questions.
CHUNK_SIZE = 1000     # max characters per chunk
CHUNK_OVERLAP = 200   # characters shared between neighbouring chunks, so an
                      # idea cut at a boundary keeps its context


def load_pdf(pdf_path: str):
    """Extract text from the PDF. Returns one Document per page.

    Each Document has:
      .page_content -> the text of that page
      .metadata     -> e.g. {"source": "file.pdf", "page": 0}  (page is 0-based)
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_into_chunks(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split pages into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split at paragraphs first, then
    lines, then words, so chunks break at natural places where possible.
    The page metadata is copied to every chunk, so we can show sources later.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def load_and_chunk(pdf_path: str):
    """Steps 1a + 1b together. Step 2 reuses this."""
    pages = load_pdf(pdf_path)
    chunks = split_into_chunks(pages)
    return pages, chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_load_and_chunk.py path/to/file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).is_file():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    pages, chunks = load_and_chunk(pdf_path)

    # --- Quality checks: is the extracted text usable? ---
    total_chars = sum(len(page.page_content) for page in pages)
    empty_pages = [page.metadata.get("page", 0) + 1 for page in pages if not page.page_content.strip()]

    print(f"Pages loaded     : {len(pages)}")
    print(f"Total characters : {total_chars}")
    if empty_pages:
        # Empty pages usually mean a scanned image with no text layer (needs OCR).
        print(f"Empty pages      : {empty_pages}  <- probably scanned images, need OCR")

    print(f"Chunks created   : {len(chunks)}  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    if chunks:
        avg = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        print(f"Average chunk    : {avg:.0f} characters")

    # --- Look at a few chunks to judge where the splits happen ---
    for i, chunk in enumerate(chunks[:3]):
        page_number = chunk.metadata.get("page", 0) + 1
        print("\n" + "=" * 70)
        print(f"Chunk {i + 1}  |  page {page_number}  |  {len(chunk.page_content)} chars")
        print("=" * 70)
        print(chunk.page_content[:500])


if __name__ == "__main__":
    main()
