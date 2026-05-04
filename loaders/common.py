from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Collapse newlines so PDF/text content embeds and chunks cleanly."""
    return " ".join(text.split()).strip()


def enrich_metadata(docs: list[Document], file_name: str) -> list[Document]:
    """Attach a stable, human-readable source filename for citations."""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["file_name"] = file_name
    return docs
