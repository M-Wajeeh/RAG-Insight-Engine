from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import EMBEDDING_MODEL
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)
