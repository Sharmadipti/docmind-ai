"""The RAG pipeline: retrieve the relevant chunks, then generate a grounded answer."""

from app.llm import NO_ANSWER, generate_answer
from app.schemas import Source
from app.vector_store import VectorStore

SNIPPET_LENGTH = 300


def answer_question(store: VectorStore, question: str, top_k: int, doc_id: str | None = None):
    """Returns (answer, sources)."""
    # 1. Retrieve: embed the question and find the closest chunks in FAISS.
    results = store.search(question, k=top_k, doc_id=doc_id)
    if not results:
        return NO_ANSWER, []

    # 2. Generate: Gemini answers from those chunks only.
    chunks = [chunk for chunk, _ in results]
    answer = generate_answer(question, chunks)

    # 3. Return the sources too, so the user can check where the answer came from.
    sources = [
        Source(
            number=number,
            document=chunk.metadata.get("source", "unknown"),
            page=chunk.metadata.get("page", 0),
            snippet=chunk.page_content[:SNIPPET_LENGTH],
            distance=round(float(distance), 4),
        )
        for number, (chunk, distance) in enumerate(results, start=1)
    ]
    return answer, sources
