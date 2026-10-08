"""Step 4: the FastAPI backend.

Endpoints:
  GET  /health      is the service running? how many documents? is Gemini configured?
  GET  /documents   list the uploaded PDFs
  POST /upload      upload a PDF -> extract, chunk, embed, store in FAISS
  POST /ask         ask a question -> retrieve chunks -> Gemini answer + sources

Run:  python -m uvicorn app.main:app --reload     then open http://localhost:8000/docs
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pypdf.errors import PdfReadError

from app.config import settings
from app.ingestion import NoTextFoundError, process_pdf
from app.llm import LLMError, LLMNotConfiguredError
from app.rag import answer_question
from app.schemas import AskRequest, AskResponse, DocumentInfo, HealthResponse, UploadResponse
from app.vector_store import VectorStore, get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docmind")

READ_BLOCK_SIZE = 1024 * 1024  # read uploads 1 MB at a time

store = VectorStore(settings.index_dir, settings.registry_path)


def get_store() -> VectorStore:
    """Dependency, so tests can swap in a store with fake embeddings."""
    return store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    # Load the embedding model once at startup, so the first request isn't slow.
    get_embeddings()
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY is not set: /upload works, but /ask will return 503.")
    yield


app = FastAPI(
    title="DocMind AI",
    description="Ask questions about your PDF documents (RAG with FAISS + Gemini).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(store: VectorStore = Depends(get_store)):
    return HealthResponse(
        status="ok",
        documents=len(store.list_documents()),
        llm_configured=bool(settings.google_api_key),
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(store: VectorStore = Depends(get_store)):
    return store.list_documents()


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf(file: UploadFile = File(...), store: VectorStore = Depends(get_store)):
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are supported.")

    data = _read_upload(file)
    # Every real PDF starts with "%PDF". This catches renamed non-PDF files.
    if not data.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is not a valid PDF.")

    doc_id = uuid.uuid4().hex
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = settings.uploads_dir / f"{doc_id}.pdf"  # our own name, never the user's (avoids path tricks)
    pdf_path.write_bytes(data)

    try:
        chunks, page_count, empty_pages = process_pdf(pdf_path, doc_id, filename)
    except NoTextFoundError as err:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err)) from err
    except PdfReadError as err:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The PDF could not be read.") from err

    record = {"doc_id": doc_id, "filename": filename, "pages": page_count, "chunks": len(chunks)}
    store.add_document(chunks, record)
    logger.info("Indexed %s: %d pages, %d chunks", filename, page_count, len(chunks))

    return UploadResponse(**record, empty_pages=empty_pages)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, store: VectorStore = Depends(get_store)):
    if not store.has_documents():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No documents yet. Upload a PDF first.")
    if request.doc_id and not store.get_document(request.doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")

    try:
        answer, sources = answer_question(
            store, request.question, request.top_k or settings.top_k, request.doc_id
        )
    except LLMNotConfiguredError as err:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err)) from err
    except LLMError as err:
        logger.error("LLM error: %s", err)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The language model request failed.") from err

    return AskResponse(answer=answer, sources=sources)


def _read_upload(file: UploadFile) -> bytes:
    """Read the upload in blocks and stop early if it's too big."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = bytearray()
    while block := file.file.read(READ_BLOCK_SIZE):
        data.extend(block)
        if len(data) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {settings.max_upload_mb} MB."
            )
    return bytes(data)
