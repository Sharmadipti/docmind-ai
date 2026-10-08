"""
DocMind AI - Step 2: Create embeddings, store them in FAISS, and search.

Embedding = a list of numbers that represents the MEANING of a text.
Texts with similar meaning get vectors that are close together, so we can
find the chunks most related to a question, even if the words differ.

Run:
  python step2_embed_and_search.py path/to/file.pdf "your question"

The first run downloads the embedding model (about 90 MB) from Hugging Face.
"""

import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from step1_load_and_chunk import load_and_chunk

# Free, local, fast model (384-dimensional vectors). Chosen in my research notes.
# The SAME model must embed both the chunks and the questions.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
TOP_K = 3   # how many chunks to retrieve per question


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index(pdf_path: str, embeddings):
    """Indexing: chunk the PDF, embed every chunk, store the vectors in FAISS."""
    _, chunks = load_and_chunk(pdf_path)
    print(f"Embedding {len(chunks)} chunks...")
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_DIR)   # saved to disk, so we don't re-embed every time
    print(f"Saved FAISS index to ./{INDEX_DIR}")
    return index


def search(index, question: str, k: int = TOP_K):
    """Querying: embed the question and return the k most similar chunks.

    FAISS returns a distance score: LOWER = MORE similar.
    """
    return index.similarity_search_with_score(question, k=k)


def main():
    if len(sys.argv) < 3:
        print('Usage: python step2_embed_and_search.py path/to/file.pdf "your question"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]
    if not Path(pdf_path).is_file():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    embeddings = get_embeddings()
    index = build_index(pdf_path, embeddings)

    print(f"\nQuestion: {question}")
    for rank, (chunk, score) in enumerate(search(index, question), start=1):
        page_number = chunk.metadata.get("page", 0) + 1
        print("\n" + "-" * 70)
        print(f"Result {rank}  |  page {page_number}  |  distance {score:.3f}")
        print("-" * 70)
        print(chunk.page_content[:400])

    # Next step (Step 3): send the question + these chunks to Gemini,
    # with a prompt that says "answer only from this context".


if __name__ == "__main__":
    main()
