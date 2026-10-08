"""Step 2: embeddings + FAISS vector index, saved to disk.

One FAISS index holds the chunks of every uploaded PDF. Each chunk's metadata has
a doc_id, so a search can be limited to one document.
"""

import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load the embedding model once and reuse it (loading takes a few seconds).

    The SAME model must embed both the chunks and the questions, otherwise the
    vectors live in different "meaning spaces" and can't be compared.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


class VectorStore:
    def __init__(self, index_dir: Path, registry_path: Path, embeddings: Embeddings | None = None):
        self.index_dir = index_dir
        self.registry_path = registry_path
        self._embeddings = embeddings
        self._index: FAISS | None = None
        self._loaded = False
        # FastAPI runs sync endpoints in a thread pool, so two uploads could run at once.
        # The lock stops them from changing the index at the same time.
        self._lock = threading.Lock()

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if (self.index_dir / "index.faiss").exists():
            # The index's docstore is saved with pickle, and loading a pickle can run code.
            # This is safe here only because this app wrote these files itself.
            self._index = FAISS.load_local(
                str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True
            )
        self._loaded = True

    def add_document(self, chunks: list[Document], record: dict) -> None:
        """Embed the chunks, add them to the index, save to disk, and register the document."""
        with self._lock:
            self._ensure_loaded()
            if self._index is None:
                self._index = FAISS.from_documents(chunks, self.embeddings)
            else:
                self._index.add_documents(chunks)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._index.save_local(str(self.index_dir))

            documents = self.list_documents()
            documents.append({**record, "uploaded_at": datetime.now(timezone.utc).isoformat()})
            self._save_registry(documents)

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Document, float]]:
        """Embed the query and return the k closest chunks with their L2 distance (lower = closer)."""
        self._ensure_loaded()
        if self._index is None:
            return []
        metadata_filter = {"doc_id": doc_id} if doc_id else None
        # With a filter, FAISS first fetches `fetch_k` candidates, then keeps the matching ones.
        return self._index.similarity_search_with_score(
            query, k=k, filter=metadata_filter, fetch_k=max(20, k * 5)
        )

    def list_documents(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def get_document(self, doc_id: str) -> dict | None:
        return next((doc for doc in self.list_documents() if doc["doc_id"] == doc_id), None)

    def has_documents(self) -> bool:
        return bool(self.list_documents())

    def _save_registry(self, documents: list[dict]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first, then swap it in, so a crash can't leave a half-written file.
        temp_path = self.registry_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(documents, indent=2), encoding="utf-8")
        temp_path.replace(self.registry_path)
