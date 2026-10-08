"""Step 1: read a PDF and split its text into overlapping chunks."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


class NoTextFoundError(Exception):
    """The PDF has no text layer, e.g. a scanned document that would need OCR."""


def load_pdf(pdf_path: Path) -> list[Document]:
    """Extract the text layer page by page. One Document per page."""
    return PyPDFLoader(str(pdf_path)).load()


def split_into_chunks(
    pages: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    """Split pages into chunks of up to `chunk_size` characters.

    The splitter tries paragraph breaks first, then lines, then spaces, so chunks
    end at natural places. Each chunk keeps its page's metadata (for sources).
    Chunks never cross pages, because each page is split on its own.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def process_pdf(pdf_path: Path, doc_id: str, filename: str) -> tuple[list[Document], int, list[int]]:
    """Load + clean metadata + chunk. Returns (chunks, page_count, empty_pages)."""
    pages = load_pdf(pdf_path)

    text_pages: list[Document] = []
    empty_pages: list[int] = []
    for page in pages:
        # PyPDF numbers pages from 0; store them from 1 so sources match what users see.
        page_number = page.metadata.get("page", 0) + 1
        if not page.page_content.strip():
            empty_pages.append(page_number)
            continue
        page.metadata = {"doc_id": doc_id, "source": filename, "page": page_number}
        text_pages.append(page)

    if not text_pages:
        raise NoTextFoundError(
            "No readable text found. This PDF may be scanned (images only), which needs OCR."
        )

    return split_into_chunks(text_pages), len(pages), empty_pages
