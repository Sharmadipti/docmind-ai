# DocMind AI: Intelligent Document Assistant

Upload PDF documents and ask questions about them in plain English. Answers come **from the documents**, with the page numbers they came from, using **RAG** (Retrieval-Augmented Generation).

**Stack:** Python · FastAPI · LangChain · Hugging Face embeddings · FAISS · Google Gemini

## How it works

```
INDEXING  (POST /upload)
PDF → extract text per page (PyPDF) → split into overlapping chunks
    → embed each chunk (all-MiniLM-L6-v2, 384 numbers) → store in FAISS → save to disk

QUERYING  (POST /ask)
question → embed (same model) → FAISS finds the top-k closest chunks
         → prompt Gemini: "answer ONLY from these passages, cite them"
         → answer + sources (document, page, snippet, distance)
```

## Project structure

```
docmind-ai/
├── app/
│   ├── config.py        # all settings, read from .env
│   ├── ingestion.py     # Step 1: PDF → text → chunks (detects scanned PDFs)
│   ├── vector_store.py  # Step 2: embeddings + FAISS index, saved to disk
│   ├── llm.py           # Step 3: grounded prompt + Gemini call
│   ├── rag.py           # retrieve → generate → sources
│   ├── schemas.py       # request/response models (Pydantic validation)
│   └── main.py          # Step 4: FastAPI endpoints
├── scripts/
│   ├── step1_load_and_chunk.py    # early exploration: inspect the chunks of a PDF
│   ├── step2_embed_and_search.py  # early exploration: try a similarity search
│   └── list_gemini_models.py      # list the Gemini models your key can use
├── tests/               # pytest: ingestion + API (fake embeddings, mocked Gemini)
├── data/                # created at runtime: uploads/, faiss_index/, documents.json
├── .env.example
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your GOOGLE_API_KEY in .env
```

## Run

```powershell
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** (Swagger UI). The first start downloads the embedding model (about 90 MB).

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Status, number of documents, whether Gemini is configured |
| GET | `/documents` | List the uploaded PDFs |
| POST | `/upload` | Upload a PDF (form field `file`) and index it |
| POST | `/ask` | `{"question": "...", "top_k": 4, "doc_id": null}` → answer + sources |

**Errors:** 400 (not a PDF, or no documents yet), 404 (unknown `doc_id`), 413 (file too large), 422 (invalid input, or a scanned PDF with no text), 502 (the Gemini request failed), 503 (`GOOGLE_API_KEY` not set).

## Tests

```powershell
python -m pytest -q
```

The tests use fake embeddings and a mocked Gemini call, so they're fast and run offline.

## Design decisions

- **Chunking:** 1,000 characters with 200 overlap, so an idea cut at a boundary isn't lost. Chunks never cross pages, so every chunk has one page number for its source.
- **Embeddings:** `all-MiniLM-L6-v2` is free, fast and runs locally. The same model embeds the chunks and the questions.
- **FAISS:** fast and in-process, with no separate server. One index holds all the documents. Each chunk has a `doc_id`, so a search can be limited to one document.
- **Grounded prompt:** Gemini must answer only from the numbered passages and cite them, or reply "I don't know…". The passages are marked as data, not instructions, which helps against prompt injection inside documents.
- **Low temperature (0.2):** more consistent, factual answers.
- **The model is loaded once at startup**, so requests are fast.
- **Resilient Gemini calls:** temporary errors (429 rate limit, 503 "high demand") are retried with exponential backoff, and every request has a 60-second timeout. A retired model name gives a clear error that says how to fix it.

## Next steps

- [ ] An evaluation set (questions with known answers) to tune the chunk size, overlap and `top_k`
- [ ] OCR for scanned PDFs
- [ ] Delete documents
- [ ] Migrate off `langchain-community` (deprecated) to the standalone integration packages
- [ ] A simple UI (Streamlit or React)
