import pytest
from langchain_core.documents import Document

from app.ingestion import NoTextFoundError, process_pdf, split_into_chunks
from tests.conftest import make_blank_pdf, make_text_pdf


def test_process_pdf_keeps_doc_id_source_and_1_based_page(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(make_text_pdf(["Refunds are processed within 14 days.", "Contact support by email."]))

    chunks, page_count, empty_pages = process_pdf(pdf_path, doc_id="abc", filename="policy.pdf")

    assert page_count == 2
    assert empty_pages == []
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert all(chunk.metadata["doc_id"] == "abc" for chunk in chunks)
    assert all(chunk.metadata["source"] == "policy.pdf" for chunk in chunks)
    assert "14 days" in chunks[0].page_content


def test_scanned_pdf_with_no_text_raises(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(make_blank_pdf())

    with pytest.raises(NoTextFoundError):
        process_pdf(pdf_path, doc_id="abc", filename="scanned.pdf")


def test_long_page_is_split_with_overlap():
    words = " ".join(f"word{i}" for i in range(400))  # about 3,000 characters
    page = Document(page_content=words, metadata={"page": 1})

    chunks = split_into_chunks([page], chunk_size=1000, chunk_overlap=200)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    # Overlap: the end of one chunk appears again at the start of the next.
    last_word_of_first = chunks[0].page_content.split()[-1]
    assert last_word_of_first in chunks[1].page_content
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
