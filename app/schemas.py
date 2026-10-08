"""Request and response shapes. Pydantic validates them automatically."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    # A whole-request example, so Swagger's "Try it out" body is ready to send.
    # Without it, Swagger fills doc_id with "string", which returns 404 "Document not found".
    model_config = ConfigDict(json_schema_extra={"examples": [{"question": "What is this document about?"}]})

    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=10, description="How many chunks to retrieve (default 4)")
    doc_id: str | None = Field(
        None, description="Only search this document (a doc_id from /upload or /documents). Leave out to search all."
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Question must have at least 3 characters")
        return value


class Source(BaseModel):
    number: int
    document: str
    page: int
    snippet: str
    distance: float = Field(description="L2 distance to the question (lower = more similar)")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    empty_pages: list[int] = Field(description="Pages with no text (probably scanned images)")


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    llm_configured: bool
