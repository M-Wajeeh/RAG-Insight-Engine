from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from retrievers.base_retriever import get_retriever
from utils.config import TOP_K


def get_mmr_retriever(db: VectorStore, k: int = TOP_K) -> VectorStoreRetriever:
    return get_retriever(db, kind="mmr", k=k)
