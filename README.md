# RAG Pipeline

A retrieval-augmented generation (RAG) pipeline that lets you ask natural-language questions over your own documents — currently PDFs (CVs, cover letters, thesis) and plain-text files. Built as a step-by-step Jupyter notebook (`notebook/RAG_Pipeline.ipynb`), with LangChain, sentence-transformers, ChromaDB, and Google Gemini.

## How it works

The notebook is organized in three stages:

```
A) DATA INGESTION                     B) QUERY RETRIEVAL            C) LLM INTEGRATION
─────────────────────────────         ───────────────────────       ─────────────────────────
PDF / TXT files
        │
        ▼
Parsing (PyMuPDF / TextLoader)
        │
        ▼
Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings (all-MiniLM-L6-v2, 384-dim)
        │
        ▼
ChromaDB vector store (collection: allpdfs)  ──────►  Query embedding  ──────►  Gemini LLM
                                                         │                        ▲
                                                    similarity search          prompt
                                                         │                        │
                                                    retrieved chunks ─────────────┘
```

1. **Parsing** — loads text files (`TextLoader`) and PDFs (`PyMuPDFLoader`), wrapped in `DirectoryLoader`.
2. **Chunking** — `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve context across chunk boundaries.
3. **Embeddings** — the local `all-MiniLM-L6-v2` model (384-dim) via sentence-transformers. Runs on your machine; no API calls.
4. **Vector store** — a persistent ChromaDB collection stores chunk text + embeddings with `hnsw:space: cosine` so the retriever can convert distance to a real similarity score (`1 - distance`). The store is wiped and rebuilt on each notebook run to stay idempotent.
5. **Retrieval** — `RAGRetriever` embeds the query, searches the store, and returns the top-K chunks filtered by a similarity threshold.
6. **Generation** — `ChatGoogleGenerativeAI` answers the question using only the retrieved context.

## Project structure

```
RAG/
├── notebook/
│   └── RAG_Pipeline.ipynb   # the whole pipeline, step by step
├── data/
│   ├── pdfs/                # drop your PDFs here
│   ├── txt_files/           # drop your .txt files here
│   └── vector_store/        # ChromaDB persistence (auto-created)
├── main.py                  # project stub (not part of the pipeline yet)
├── pyproject.toml           # uv project + dependencies
├── requirements.txt         # pip alternative
└── uv.lock
```

## Prerequisites

- Python **3.14** (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Gemini API key (free) from [Google AI Studio](https://aistudio.google.com/)

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python -m pip install -r requirements.txt
```

## API key setup

Create a `.env` file in the project root (it's git-ignored) and add your Gemini key:

```
GEMINI_API_KEY=your_key_here
```

> **Note:** if you previously set up the key as `GOOGLE_API_KEY`, the notebook currently reads `GEMINI_API_KEY` — rename it in `.env` or update the notebook.

## Usage

1. Put your documents in `data/pdfs/` and/or `data/txt_files/`.
2. Launch the notebook:

   ```bash
   jupyter notebook notebook/RAG_Pipeline.ipynb
   ```

3. Run the cells **in order**:
   - **Ingestion** — loads, chunks, and embeds every file in the data folders.
   - **Vector database** — initializes ChromaDB and adds the chunks.
   - **Retrieval** — build a `RAGRetriever` and test queries.
   - **LLM integration** — ask questions and get grounded answers:

     ```python
     answer = rag_simple("What is abu bakar farooq's professional experience?", rag_retriever, llm)
     print(answer)
     ```

### Configuration

| Setting | Location | Default |
|---|---|---|
| Chunk size / overlap | `split_documents()` | 1000 / 200 |
| Embedding model | `EmbeddingManager.__init__` | `all-MiniLM-L6-v2` |
| Vector store dir | `VectorDatabase.__init__` | `data/vector_store` |
| Collection name | `VectorDatabase.__init__` | `allpdfs` |
| Search model | `MODEL_NAME` in the LLM cell | `gemini-3.5-flash` |
| `top_k` / threshold | `rag_retriever.retrieve(...)` | 5 / 0.1 |

## Known issues

- **Free-tier Gemini quota:** the free tier allows **~20 generate-content requests/day per model**. The error `429 RESOURCE_EXHAUSTED` means you've hit today's cap — it resets at midnight Pacific. Each model has its own bucket, so switching `MODEL_NAME` (e.g. to a different Gemini variant) gives a fresh 20. The ingestion/retrieval stages (embeddings + ChromaDB) run entirely locally and are not subject to this.

## Notes

- The embeddings and retrieval stages never leave your machine — only the final generation step calls the Gemini API.
- Want **unlimited free generation**? Swap the Gemini call for a locally hosted model, e.g. LangChain's `ChatOllama` with a Llama/Qwen model. The rest of the pipeline works unchanged.
