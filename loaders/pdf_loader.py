import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from utils.logger import setup_logger
load_dotenv()

logger = setup_logger(__name__)


def clean_text(text):
    return text.replace("\n", " ").strip()


def load_pdfs(folder_path):
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)

            try:
                loader = PyPDFLoader(path)
                docs = loader.load()

                for doc in docs:
                    # clean content
                    doc.page_content = clean_text(doc.page_content)

                    # enrich metadata
                    doc.metadata["file_name"] = file

                logger.info(f"Loaded {file} ({len(docs)} pages)")
                all_docs.extend(docs)

            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

    logger.info(f"\nTotal documents loaded: {len(all_docs)}")

    # preview (VERY IMPORTANT for debugging)
    for i, doc in enumerate(all_docs[:2]):
        print(f"\n--- Preview {i} ---")
        print(f"File: {doc.metadata.get('file_name')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:300])

    return all_docs

if __name__ == "__main__":
    docs = load_pdfs("data/raw")
    logger.info(f"Total documents loaded: {len(docs)}")