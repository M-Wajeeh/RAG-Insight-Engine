import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from loaders.common import enrich_metadata
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_pdfs(data_path: str) -> list[Document]:
    if not os.path.isdir(data_path):
        logger.warning("PDF data path does not exist: %s", data_path)
        return []

    all_docs: list[Document] = []

    for file in sorted(os.listdir(data_path)):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(data_path, file)
        try:
            docs = PyPDFLoader(path).load()
        except Exception as exc:
            logger.error("Failed to load %s: %s", file, exc)
            continue

        all_docs.extend(enrich_metadata(docs, file))
        logger.info("Loaded PDF %s (%d pages)", file, len(docs))

    logger.info("Total PDF documents loaded: %d", len(all_docs))
    return all_docs
