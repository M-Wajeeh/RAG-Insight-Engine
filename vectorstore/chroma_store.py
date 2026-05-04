import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.config import CHROMA_DB_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _has_existing_index(persist_directory: str) -> bool:
    return os.path.isdir(persist_directory) and bool(os.listdir(persist_directory))


def create_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    if not chunks:
        raise ValueError("Cannot create vector store: no chunks provided.")

    logger.info("Creating Chroma vector store at %s (%d chunks)", persist_directory, len(chunks))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def load_vectorstore(
    embeddings: Embeddings,
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    logger.info("Loading Chroma vector store from %s", persist_directory)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def get_or_create_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    """Reuse an existing index when present, otherwise build a new one.

    Avoids recomputing embeddings across runs.
    """
    if _has_existing_index(persist_directory):
        return load_vectorstore(embeddings, persist_directory)
    return create_vectorstore(chunks, embeddings, persist_directory)
