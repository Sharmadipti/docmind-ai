from types import SimpleNamespace

import pytest
from google.genai import errors
from langchain_core.documents import Document

from app import llm

CHUNKS = [Document(page_content="Refunds take 14 days.", metadata={"source": "policy.pdf", "page": 1})]


def fake_client(generate_content):
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_retired_model_gives_clear_error(monkeypatch):
    def not_found(**kwargs):
        raise errors.APIError(404, {"error": {"code": 404, "message": "gone", "status": "NOT_FOUND"}})

    monkeypatch.setattr(llm, "get_client", lambda: fake_client(not_found))

    with pytest.raises(llm.LLMError, match="GEMINI_MODEL"):
        llm.generate_answer("How long do refunds take?", CHUNKS)


def test_empty_response_raises(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: fake_client(lambda **kwargs: SimpleNamespace(text="")))

    with pytest.raises(llm.LLMError, match="empty"):
        llm.generate_answer("How long do refunds take?", CHUNKS)


def test_prompt_numbers_passages_with_source_and_page():
    prompt = llm.build_prompt("How long?", CHUNKS)

    assert "[1] (source: policy.pdf, page 1)" in prompt
    assert prompt.endswith("Question: How long?")
