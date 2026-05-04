# RAG Insight Engine

Modular Retrieval-Augmented Generation (RAG) system for asking grounded questions
over your own documents.
The project includes pluggable loaders, recursive chunking, HuggingFace embeddings,
a persistent Chroma vector store, two retrievers (similarity and MMR), a strict
citation-aware Groq LLM generator, a CLI, a Streamlit UI, and an evaluation
harness for comparing retrieval strategies.

## Features

- Pluggable loaders for PDF and plain text with shared cleanup and metadata enrichment
- Recursive chunking with configurable size and overlap
- HuggingFace sentence-transformer embeddings (MiniLM by default; MPNet supported)
- Persistent Chroma vector store; embeddings reused across runs
- Two retrievers out of the box: similarity and MMR (diverse)
- Strict, citation-aware prompt grounded in document content **and** metadata
- Stable structured output: every answer comes back as `{"answer": ..., "sources": [...]}`
- Minimal CLI entry point (`app.py`)
- Streamlit UI with retriever and `top_k` controls, source rendering, and raw JSON view
- Evaluation harness (`evaluate_rag.py`) for sweeping chunk sizes, embeddings, and retrievers

## Project Structure

```text
RAG-Insight-Engine/
├── data/
│   └── raw/                      # Drop your .pdf and .txt files here
├── chroma_db/                    # Persistent Chroma index (auto-created, gitignored)
├── loaders/
│   ├── common.py                 # clean_text + metadata enrichment
│   ├── pdf_loader.py             # PyPDFLoader-based PDF ingestion
│   └── text_loader.py            # TextLoader-based .txt ingestion
├── splitters/
│   └── chunker.py                # RecursiveCharacterTextSplitter wrapper
├── embeddings/
│   └── embedder.py               # HuggingFaceEmbeddings wrapper
├── vectorstore/
│   └── chroma_store.py           # Chroma create / load / get-or-create
├── retrievers/
│   ├── base_retriever.py         # Unified get_retriever(kind, k)
│   └── mmr_retriever.py          # Thin compatibility wrapper
├── llm/
│   └── generator.py              # Groq ChatGroq + grounded prompt + RAGAnswer
├── pipeline/
│   └── rag_pipeline.py           # End-to-end RAGPipeline orchestrator
├── utils/
│   ├── config.py                 # Single source of truth for all knobs
│   └── logger.py                 # Idempotent stdout logger
├── app.py                        # Minimal CLI
├── streamlit_app.py              # Streamlit UI
├── evaluate_rag.py               # Retrieval / generation evaluation harness
├── pyrightconfig.json
├── requirements.txt
├── .env.example
└── README.md
```

## Tech Stack

- Python 3.11+
- LangChain (`langchain`, `langchain-community`, `langchain-core`, `langchain-text-splitters`)
- LangChain integrations: `langchain-chroma`, `langchain-huggingface`, `langchain-groq`
- ChromaDB (persistent vector store)
- HuggingFace `sentence-transformers` (MiniLM, MPNet)
- Groq Chat API (`llama-3.3-70b-versatile` by default)
- Streamlit (UI)
- pypdf (PDF parsing)

## Local Setup

### 1) Clone and install

```powershell
git clone https://github.com/M-Wajeeh/RAG-Insight-Engine.git
cd RAG-Insight-Engine

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure secrets

```powershell
copy .env.example .env
# Linux/macOS: cp .env.example .env
```

Edit `.env` and set your Groq key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Get one from https://console.groq.com/.

### 3) Add documents

Drop the PDFs and `.txt` files you want to query into `data/raw/`.

### 4) Run the CLI

```powershell
python app.py
```

The first run will:

1. Load documents from `data/raw/`.
2. Chunk and embed them.
3. Persist a Chroma index in `chroma_db/`.
4. Print the answer plus its sources as JSON.

Subsequent runs reuse the persisted index, so they start fast.

### 5) Run the Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

Dashboard: http://localhost:8501

### 6) Run the evaluation harness

Compare retrievers, chunk sizes, and embeddings against a fixed query set.

```powershell
# Retrieval-only sweep across chunk sizes (fast, no LLM calls)
python evaluate_rag.py --queries 6 --chunk-sizes 300 500 800 `
    --embeddings minilm --retrievers similarity mmr --top-k 4 --skip-answers

# Full sweep including LLM-generated answers and grounding checks
python evaluate_rag.py --queries 6 --chunk-sizes 500 `
    --embeddings minilm mpnet --retrievers similarity mmr --top-k 4
```

For each `(chunk_size, embedding)` configuration the script prints:
- retrieved chunks per retriever
- generated answer (when not `--skip-answers`)
- cited sources
- grounding signals (fallback used? duplicate sources?)

## CLI Output Format

`RAGPipeline.ask(query)` and `generate_answer(...)` always return the same shape:

```python
{
    "answer": "...",
    "sources": [
        {"file_name": "NIPS-2017.pdf", "page": 0, "title": "Attention is All you Need"},
        ...
    ],
}
```

Example CLI run:

```text
=== QUERY ===
Tell about Table 1

=== ANSWER ===
Table 1 reports maximum path lengths, per-layer complexity, and minimum number
of sequential operations for self-attention, recurrent, convolutional, and
restricted self-attention layers [Source 1].

=== SOURCES ===
[
  {"file_name": "NIPS-2017.pdf", "page": 5, "title": "Attention is All you Need"},
  ...
]
```

The prompt itself enforces strict grounding: if no field in the retrieved
context (content, title, authors, abstract, page) supports the answer, the
model returns the exact string `Not found in the provided documents.` instead
of hallucinating.

## Streamlit Dashboard

The Streamlit UI includes:

- Retriever selector (`similarity` or `mmr`)
- `top_k` slider (1–20)
- Question input
- Answer panel
- Sources panel with file name, page, and title
- Expander with the raw structured `RAGAnswer` JSON

The pipeline (embeddings + vector store + LLM client) is cached per
`(retriever_kind, top_k)` via `st.cache_resource`, so the heavy work happens
once instead of on every click.

## Configuration

All knobs live in `utils/config.py`:

| Setting                  | Default                                      | Notes                              |
| ------------------------ | -------------------------------------------- | ---------------------------------- |
| `DATA_PATH`              | `data/raw`                                   | Document source folder             |
| `CHROMA_DB_DIR`          | `chroma_db`                                  | Persisted vector store             |
| `CHUNK_SIZE`             | `500`                                        | Characters per chunk               |
| `CHUNK_OVERLAP`          | `50`                                         |                                    |
| `TOP_K`                  | `4`                                          | Retrieved chunks per query         |
| `MMR_FETCH_K_MULTIPLIER` | `2`                                          | `fetch_k = TOP_K * multiplier`     |
| `DEFAULT_RETRIEVER`      | `similarity`                                 |                                    |
| `EMBEDDING_MODEL`        | `sentence-transformers/all-MiniLM-L6-v2`     | Override per call if needed        |
| `GROQ_MODEL`             | `llama-3.3-70b-versatile`                    | Any current Groq chat model        |
| `TEMPERATURE`            | `0`                                          | Lower = more grounded              |

## How It Works

1. **Load** — `loaders/` reads PDFs and `.txt` files from `data/raw/`, cleans
   whitespace, and tags each document with `file_name` metadata.
2. **Chunk** — `splitters/chunker.py` recursively splits documents into
   ~500-char chunks with 50-char overlap.
3. **Embed** — `embeddings/embedder.py` runs each chunk through a HuggingFace
   sentence-transformer (`MiniLM` by default).
4. **Store** — `vectorstore/chroma_store.py` writes a persistent Chroma index.
   On subsequent runs it loads the existing index instead of recomputing
   embeddings.
5. **Retrieve** — `retrievers/base_retriever.py` exposes `get_retriever(kind, k)`
   for `similarity` or `mmr` search.
6. **Generate** — `llm/generator.py` formats retrieved chunks and their
   metadata into a strict prompt, calls Groq, and returns
   `RAGAnswer = {"answer": str, "sources": list[Source]}`.
7. **Orchestrate** — `pipeline/rag_pipeline.py` wires steps 1–6 into a single
   `RAGPipeline.ask(query)` call. `app.py` and `streamlit_app.py` are thin
   front-ends over this pipeline.

## Evaluation

`evaluate_rag.py` is the canonical way to measure changes. It supports:

- **Retriever comparison** — `--retrievers similarity mmr` runs both against
  the same vector store for fair comparison.
- **Chunk-size sweep** — `--chunk-sizes 300 500 800` rebuilds the index per
  size and re-runs the same queries.
- **Embedding comparison** — `--embeddings minilm mpnet` tests how a stronger
  embedding model affects retrieval.
- **Retrieval-only mode** — `--skip-answers` avoids LLM calls when you only
  care about which chunks come back.
- **Grounding signals** — for every answered query, the script prints whether
  the model fell back to the "not found" string and whether the retrieved
  chunks repeat sources.

The default test query set covers common research-paper questions:

```python
TEST_QUERIES = [
    "What is the main contribution?",
    "What problem does the paper solve?",
    "Explain the methodology",
    "What are the limitations?",
    "Summarize the paper",
    "Tell about Table 1",
]
```

## Troubleshooting

### Streamlit shows `Not found in the provided documents.` for everything

- Make sure your documents actually live in `data/raw/`.
- Bump `TOP_K` (sidebar slider) to 8–10 — small `k` plus large documents
  often misses the right chunk.
- Try the `mmr` retriever for broader coverage.

### Want a clean rebuild of the vector store

Delete the persisted index:

```powershell
Remove-Item -Recurse -Force chroma_db
```

Next run rebuilds it from scratch using the current chunking and embedding
settings.

### Switching embedding model

Always rebuild the index after changing `EMBEDDING_MODEL` (or pass a different
`--embeddings` flag to the evaluator). Mixing vectors from different models
in the same Chroma collection produces unusable results.

### Groq model errors (`model has been decommissioned`)

Update `GROQ_MODEL` in `utils/config.py` to a current model
(see https://console.groq.com/docs/deprecations).

## License

MIT — see `LICENSE` for details.

## About

Modular RAG system focused on correctness, grounding, and debuggability. Built
with LangChain, ChromaDB, HuggingFace embeddings, and the Groq LLM API. Comes
with a Streamlit UI and an evaluation harness so you can measure changes
instead of guessing.
