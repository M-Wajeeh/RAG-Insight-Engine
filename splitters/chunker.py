from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.pdf_loader import load_pdfs
from loaders.text_loader import load_texts
from utils.config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_documents(data_path: str) -> list[Document]:
    """Load all supported file types from `data_path`."""
    docs: list[Document] = []
    docs.extend(load_pdfs(data_path))
    docs.extend(load_texts(data_path))

    logger.info("Loaded %d documents from %s", len(docs), data_path)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if not docs:
        logger.warning("split_documents called with no documents")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    logger.info(
        "Created %d chunks (chunk_size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
