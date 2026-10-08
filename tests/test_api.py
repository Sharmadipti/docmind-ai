from app.llm import LLMError, LLMNotConfiguredError
from tests.conftest import make_blank_pdf, make_text_pdf

TEXT_PDF = make_text_pdf(["Refunds are processed within 14 days.", "Contact support by email."])


def upload(client, content: bytes, filename: str = "policy.pdf"):
    return client.post("/upload", files={"file": (filename, content, "application/pdf")})


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["documents"] == 0


def test_upload_indexes_pdf_and_lists_it(client):
    res = upload(client, TEXT_PDF)

    assert res.status_code == 201
    body = res.json()
    assert body["filename"] == "policy.pdf"
    assert body["pages"] == 2
    assert body["chunks"] == 2

    docs = client.get("/documents").json()
    assert [doc["doc_id"] for doc in docs] == [body["doc_id"]]


def test_upload_rejects_non_pdf_filename(client):
    assert upload(client, b"hello", filename="notes.txt").status_code == 400


def test_upload_rejects_renamed_non_pdf(client):
    assert upload(client, b"not really a pdf", filename="fake.pdf").status_code == 400


def test_upload_scanned_pdf_returns_clear_422(client):
    res = upload(client, make_blank_pdf(), filename="scanned.pdf")
    assert res.status_code == 422
    assert "scanned" in res.json()["detail"]


def test_ask_before_any_upload_returns_400(client):
    res = client.post("/ask", json={"question": "What is the refund policy?"})
    assert res.status_code == 400


def test_ask_validates_question(client):
    assert client.post("/ask", json={"question": "  a "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_answer_with_sources(client, monkeypatch):
    upload(client, TEXT_PDF)
    captured = {}

    def fake_generate(question, chunks):
        captured["question"] = question
        captured["chunks"] = chunks
        return "Refunds take 14 days [1]."

    monkeypatch.setattr("app.rag.generate_answer", fake_generate)

    res = client.post("/ask", json={"question": "How long do refunds take?", "top_k": 2})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Refunds take 14 days [1]."
    assert len(body["sources"]) == 2
    assert {source["page"] for source in body["sources"]} == {1, 2}
    assert captured["question"] == "How long do refunds take?"
    assert len(captured["chunks"]) == 2


def test_ask_unknown_doc_id_returns_404(client):
    upload(client, TEXT_PDF)
    res = client.post("/ask", json={"question": "Anything?", "doc_id": "missing"})
    assert res.status_code == 404


def test_ask_without_api_key_returns_503(client, monkeypatch):
    upload(client, TEXT_PDF)

    def not_configured(question, chunks):
        raise LLMNotConfiguredError("GOOGLE_API_KEY is not set.")

    monkeypatch.setattr("app.rag.generate_answer", not_configured)
    assert client.post("/ask", json={"question": "Refund policy?"}).status_code == 503


def test_ask_when_gemini_fails_returns_502(client, monkeypatch):
    upload(client, TEXT_PDF)

    def failing(question, chunks):
        raise LLMError("boom")

    monkeypatch.setattr("app.rag.generate_answer", failing)
    assert client.post("/ask", json={"question": "Refund policy?"}).status_code == 502
