import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from loaders.common import enrich_metadata
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_texts(data_path: str) -> list[Document]:
    if not os.path.isdir(data_path):
        logger.warning("Text data path does not exist: %s", data_path)
        return []

    all_docs: list[Document] = []

    for file in sorted(os.listdir(data_path)):
        if not file.lower().endswith(".txt"):
            continue

        path = os.path.join(data_path, file)
        try:
            docs = TextLoader(path, encoding="utf-8").load()
        except Exception as exc:
            logger.error("Failed to load %s: %s", file, exc)
            continue

        all_docs.extend(enrich_metadata(docs, file))
        logger.info("Loaded text %s (%d documents)", file, len(docs))

    logger.info("Total text documents loaded: %d", len(all_docs))
    return all_docs
