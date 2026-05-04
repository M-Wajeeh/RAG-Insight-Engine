import argparse
import json
import tempfile
from dataclasses import dataclass
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings.embedder import get_embeddings
from llm.generator import generate_answer, get_llm
from retrievers.base_retriever import get_retriever
from splitters.chunker import load_documents, split_documents
from utils.config import DATA_PATH, TOP_K
from utils.logger import setup_logger

logger = setup_logger(__name__)


TEST_QUERIES = [
    "What is the main contribution?",
    "What problem does the paper solve?",
    "Explain the methodology",
    "What are the limitations?",
    "Summarize the paper",
    "Tell about Table 1",
]


EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


@dataclass(frozen=True)
class ExperimentConfig:
    chunk_size: int
    embedding_label: str
    embedding_model: str
    top_k: int
    include_answers: bool


def _build_vectorstore(
    documents: list[Document],
    chunk_size: int,
    embedding_model: str,
) -> Chroma:
    chunks = split_documents(documents, chunk_size=chunk_size)
    embeddings = get_embeddings(embedding_model)

    persist_dir = tempfile.mkdtemp(prefix="rag_eval_chroma_")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def _format_source(doc: Document) -> str:
    meta = doc.metadata or {}
    file_name = meta.get("file_name") or meta.get("source", "unknown")
    page = meta.get("page", "N/A")
    return f"{file_name} | page {page}"


def _print_chunks(docs: list[Document]) -> None:
    seen = set()
    for index, doc in enumerate(docs, start=1):
        source = _format_source(doc)
        seen.add(source)
        preview = " ".join(doc.page_content.split())[:300]
        print(f"  [{index}] {source}")
        print(f"      {preview}")
    print(f"  Unique sources: {len(seen)}")


def _print_grounding(answer: str, docs: list[Document]) -> None:
    fallback = "not found in the provided documents" in answer.lower()
    repeated = len({_format_source(d) for d in docs}) < len(docs)
    print(
        f"  Grounding: fallback={fallback}, "
        f"chunks={len(docs)}, repeated_sources={repeated}"
    )


def run_experiment(
    config: ExperimentConfig,
    documents: list[Document],
    queries: Iterable[str],
    retriever_kinds: list[str],
) -> None:
    print("\n" + "=" * 100)
    print(
        f"Experiment: chunk_size={config.chunk_size} "
        f"embedding={config.embedding_label} top_k={config.top_k}"
    )
    print("=" * 100)

    db = _build_vectorstore(documents, config.chunk_size, config.embedding_model)
    llm = get_llm() if config.include_answers else None

    for query in queries:
        print(f"\nQuery: {query}")
        for kind in retriever_kinds:
            print(f"\n Retriever: {kind}")
            retriever = get_retriever(db, kind=kind, k=config.top_k)
            docs = retriever.invoke(query)
            _print_chunks(docs)

            if llm is not None:
                result = generate_answer(query, docs, llm)
                print(f"  Answer: {result['answer']}")
                print(f"  Sources: {json.dumps(result['sources'], default=str)}")
                _print_grounding(result["answer"], docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and answer quality across "
        "retriever, chunk, and embedding settings."
    )
    parser.add_argument("--queries", type=int, default=3,
                        help="Number of test queries to run (max 6).")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500])
    parser.add_argument("--embeddings", nargs="+",
                        choices=list(EMBEDDING_MODELS.keys()), default=["minilm"])
    parser.add_argument("--retrievers", nargs="+",
                        choices=["similarity", "mmr"], default=["similarity", "mmr"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--skip-answers", action="store_true",
                        help="Skip LLM answer generation. Useful for retrieval-only sweeps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(DATA_PATH)
    queries = TEST_QUERIES[: args.queries]

    for chunk_size in args.chunk_sizes:
        for label in args.embeddings:
            config = ExperimentConfig(
                chunk_size=chunk_size,
                embedding_label=label,
                embedding_model=EMBEDDING_MODELS[label],
                top_k=args.top_k,
                include_answers=not args.skip_answers,
            )
            run_experiment(config, documents, queries, args.retrievers)


if __name__ == "__main__":
    main()
