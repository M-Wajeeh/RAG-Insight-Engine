import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from utils import config
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

def clean_text(text):
    return text.replace("\n", " ").strip()


def safe_preview(text, limit=300):
    return text[:limit].encode("ascii", errors="replace").decode("ascii")


def load_text(DATA_PATH):
    all_docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):  # Changed extension to .txt
            path = os.path.join(DATA_PATH, file)

            try:
                loader = TextLoader(path, encoding="utf-8") 
                docs = loader.load()

                for doc in docs:
                    # clean content
                    doc.page_content = clean_text(doc.page_content)

                    # enrich metadata
                    doc.metadata["file_name"] = file

                
                logger.info(f"Loaded {file} (1 document)")
                all_docs.extend(docs)

            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

    logger.info(f"\nTotal text documents loaded: {len(all_docs)}")


    for i, doc in enumerate(all_docs[:2]):
        print(f"\n--- Preview {i} ---")
        print(f"File: {doc.metadata.get('file_name')}")
        # Note: TextLoader doesn't auto-generate 'page' metadata, so this will print 'None'
        print(f"Page: {doc.metadata.get('page', 'N/A')}") 
        print(safe_preview(doc.page_content))

    return all_docs

if __name__ == "__main__":
    docs = load_text(config.DATA_PATH) # This now matches the function name!
    logger.info(f"Total documents loaded: {len(docs)}")