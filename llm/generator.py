from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from utils.config import GROQ_MODEL, TEMPERATURE
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Source(TypedDict):
    file_name: str | None
    page: int | str | None
    title: str | None


class RAGAnswer(TypedDict):
    answer: str
    sources: list[Source]


PROMPT_TEMPLATE = """You are a precise research assistant.

Rules:
1. Answer ONLY using the context below. The context includes both document text (Content)
   and document metadata (Title, Authors, Abstract, File, Page). Both are valid evidence.
2. If the answer is not supported by any field in the context, reply exactly:
   "Not found in the provided documents."
3. Do not invent facts, numbers, citations, or authors that are not in the context.
4. Cite supporting passages inline as [Source N], where N is the source number from the context.
5. Be concise.

Context:
{context}

Question: {query}

Answer:"""


def get_llm() -> BaseChatModel:
    logger.info("Loading LLM: %s", GROQ_MODEL)
    return ChatGroq.model_validate(
        {"model": GROQ_MODEL, "temperature": TEMPERATURE}
    )


def _format_doc(index: int, doc: Document) -> str:
    meta = doc.metadata or {}
    lines = [
        f"[Source {index}]",
        f"File: {meta.get('file_name', 'unknown')}",
        f"Page: {meta.get('page', 'N/A')}",
    ]

    for label, key in (
        ("Title", "title"),
        ("Authors", "author"),
        ("Abstract", "description-abstract"),
    ):
        value = meta.get(key)
        if value:
            lines.append(f"{label}: {value}")

    lines.append(f"Content: {doc.page_content.strip()}")
    return "\n".join(lines)


def _build_sources(docs: list[Document]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[object, object, object]] = set()
    for doc in docs:
        meta = doc.metadata or {}
        key = (meta.get("file_name"), meta.get("page"), meta.get("title"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "title": meta.get("title"),
            }
        )
    return sources


def _extract_text(response_content) -> str:
    if isinstance(response_content, str):
        return response_content
    if isinstance(response_content, list):
        parts = [
            part if isinstance(part, str) else part.get("text", "")
            for part in response_content
        ]
        return "".join(parts)
    return str(response_content)


def generate_answer(
    query: str,
    docs: list[Document],
    llm: BaseChatModel,
) -> RAGAnswer:
    """Generate a grounded answer plus its source documents.

    Always returns a stable ``{"answer": ..., "sources": [...]}`` shape so callers
    can render or persist results consistently.
    """
    if not docs:
        logger.warning("generate_answer called with no retrieved documents")
        return {"answer": "Not found in the provided documents.", "sources": []}

    context = "\n\n".join(_format_doc(i, doc) for i, doc in enumerate(docs, start=1))
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = llm.invoke(prompt)
    answer = _extract_text(response.content).strip()

    return {"answer": answer, "sources": _build_sources(docs)}
