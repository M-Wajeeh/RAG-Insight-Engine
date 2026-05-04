from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from embeddings.embedder import get_embeddings
from llm.generator import RAGAnswer, generate_answer, get_llm
from retrievers.base_retriever import get_retriever
from splitters.chunker import load_documents, split_documents
from utils.config import DATA_PATH, DEFAULT_RETRIEVER, TOP_K
from utils.logger import setup_logger
from vectorstore.chroma_store import get_or_create_vectorstore

logger = setup_logger(__name__)


class RAGPipeline:
    """End-to-end RAG: ingest once, retrieve, generate grounded answers."""

    def __init__(
        self,
        retriever_kind: str = DEFAULT_RETRIEVER,
        top_k: int = TOP_K,
        data_path: str = DATA_PATH,
    ) -> None:
        logger.info("Initializing RAG pipeline (retriever=%s, k=%d)", retriever_kind, top_k)

        self.embeddings = get_embeddings()
        self.llm: BaseChatModel = get_llm()

        chunks = self._prepare_chunks(data_path)
        self.db: VectorStore = get_or_create_vectorstore(chunks, self.embeddings)
        self.retriever: VectorStoreRetriever = get_retriever(
            self.db, kind=retriever_kind, k=top_k
        )

    @staticmethod
    def _prepare_chunks(data_path: str) -> list[Document]:
        """Load + split docs only when needed; reused if vector DB already exists."""
        docs = load_documents(data_path)
        return split_documents(docs)

    def retrieve(self, query: str) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        docs = self.retriever.invoke(query)
        logger.info("Retrieved %d chunks for query: %s", len(docs), query)
        for i, doc in enumerate(docs, start=1):
            preview = " ".join(doc.page_content.split())[:140]
            logger.debug("[chunk %d] %s", i, preview)
        return docs

    def ask(self, query: str) -> RAGAnswer:
        docs = self.retrieve(query)
        result = generate_answer(query, docs, self.llm)
        logger.info("Answer generated (chars=%d, sources=%d)",
                    len(result["answer"]), len(result["sources"]))
        return result
