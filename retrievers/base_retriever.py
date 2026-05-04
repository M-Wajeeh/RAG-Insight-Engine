from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from utils.config import MMR_FETCH_K_MULTIPLIER, TOP_K
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_retriever(
    db: VectorStore,
    kind: str = "similarity",
    k: int = TOP_K,
) -> VectorStoreRetriever:
    """Return a retriever configured for `similarity` or `mmr` search."""
    if kind == "similarity":
        logger.info("Initializing similarity retriever (k=%d)", k)
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    if kind == "mmr":
        fetch_k = k * MMR_FETCH_K_MULTIPLIER
        logger.info("Initializing MMR retriever (k=%d, fetch_k=%d)", k, fetch_k)
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )

    raise ValueError(f"Unknown retriever kind: {kind!r}. Use 'similarity' or 'mmr'.")


def get_similarity_retriever(db: VectorStore, k: int = TOP_K) -> VectorStoreRetriever:
    return get_retriever(db, kind="similarity", k=k)
