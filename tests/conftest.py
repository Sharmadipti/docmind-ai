import io

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter

from app.main import app, get_store
from app.vector_store import VectorStore


def make_text_pdf(page_texts: list[str]) -> bytes:
    """Build a small, valid PDF with one line of real text per page (no extra libraries)."""
    objects: list[bytes] = [b"", b""]  # 1 = catalog, 2 = pages tree; filled in at the end
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_id = len(objects)
    page_ids = []
    for text in page_texts:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_id = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        page_ids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    return bytes(out)


def make_blank_pdf(pages: int = 2) -> bytes:
    """A PDF with pages but no text at all, like a scanned document."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    # Fake embeddings: fast and offline, so tests don't download the real model.
    return VectorStore(
        tmp_path / "faiss_index",
        tmp_path / "documents.json",
        embeddings=DeterministicFakeEmbedding(size=384),
    )


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.settings", _settings_with_data_dir(tmp_path))
    app.dependency_overrides[get_store] = lambda: store
    # Not using "with TestClient(...)" on purpose: that would run the startup code,
    # which loads the real embedding model.
    yield TestClient(app)
    app.dependency_overrides.clear()


def _settings_with_data_dir(data_dir):
    from dataclasses import replace

    from app.config import settings

    return replace(settings, data_dir=data_dir)
