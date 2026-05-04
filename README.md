# RAG Insight Engine

A modular Retrieval-Augmented Generation (RAG) system for asking grounded
questions about your own documents.

Built with **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and the
**Groq** LLM API. Comes with a CLI, a Streamlit UI, and an evaluation harness
for comparing retrieval strategies.

## Features

- Pluggable loaders (PDF + plain text) with shared cleanup and metadata enrichment.
- Recursive chunking with configurable size/overlap.
- HuggingFace sentence-transformer embeddings (MiniLM by default; MPNet supported).
- Persistent Chroma vector store; embeddings are reused across runs.
- Two retrievers out of the box: similarity and MMR (diverse).
- Strict, citation-aware prompt that surfaces answers grounded in document
  content **and** metadata (title, authors, abstract).
- Structured output: every answer comes back as `{"answer": ..., "sources": [...]}`.
- Streamlit UI for interactive querying.
- Evaluation script for sweeping chunk sizes, embeddings, and retrievers.

## Project layout

```text
loaders/        # PDF + text loaders, shared helpers
splitters/      # Recursive chunker
embeddings/     # HuggingFace embedding wrapper
vectorstore/    # Chroma store: load / create / load-or-create
retrievers/     # Unified get_retriever() (similarity, mmr)
llm/            # Groq LLM wrapper + grounded answer generator
pipeline/       # End-to-end RAGPipeline
utils/          # Config + logger
app.py          # Minimal CLI entry point
streamlit_app.py# Streamlit UI
evaluate_rag.py # Retrieval / generation evaluation harness
data/raw/       # Drop your documents here (.pdf or .txt)
```

## Requirements

- Python 3.11+
- A Groq API key

## Setup

```powershell
git clone <your-repo-url>
cd RAG-Insight-Engine

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

copy .env.example .env         # Linux/macOS: cp .env.example .env
# then edit .env and set GROQ_API_KEY=...
```

Drop the documents you want to query into `data/raw/`.

## Run the CLI

```powershell
python app.py
```

The first run will:

1. Load documents from `data/raw/`.
2. Chunk and embed them.
3. Persist a Chroma index in `chroma_db/` so subsequent runs are fast.
4. Print the answer plus its sources as JSON.

## Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

Open http://localhost:8501. Use the sidebar to switch between similarity and
MMR retrievers and tune `top_k`. The pipeline is cached per
`(retriever, top_k)` so embeddings/DB are not rebuilt on every click.

## Run the evaluation harness

Compare retrievers, chunk sizes, and embeddings against a fixed query set:

```powershell
# Retrieval-only (fast) sweep across chunk sizes
python evaluate_rag.py --queries 6 --chunk-sizes 300 500 800 ^
    --embeddings minilm --retrievers similarity mmr --top-k 4 --skip-answers

# Full sweep with LLM-generated answers and grounding checks
python evaluate_rag.py --queries 6 --chunk-sizes 500 ^
    --embeddings minilm mpnet --retrievers similarity mmr --top-k 4
```

For each `(chunk_size, embedding)` configuration, the script prints the
retrieved chunks per retriever, the generated answer, the cited sources, and
basic grounding signals (fallback used? duplicate sources?).

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

## Output format

`RAGPipeline.ask(query)` and `generate_answer(...)` always return:

```python
{
    "answer": "...",
    "sources": [
        {"file_name": "...", "page": ..., "title": "..."},
        ...
    ],
}
```

The prompt itself enforces strict grounding: if the answer is not supported by
the retrieved context (content **or** metadata), the model returns the exact
string `Not found in the provided documents.` instead of hallucinating.

## Notes

- The vector store is reused if it already exists in `CHROMA_DB_DIR`. Delete
  that folder if you change embedding models or chunking and want a clean
  rebuild.
- Don't commit `chroma_db/`, `.env`, or model caches — the included
  `.gitignore` already covers them.
