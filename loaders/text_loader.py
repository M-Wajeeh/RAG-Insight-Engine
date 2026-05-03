import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

def clean_text(text):
    return text.replace("\n", " ").strip()


def load_text(folder_path):
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):  # Changed extension to .txt
            path = os.path.join(folder_path, file)

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
        print(doc.page_content[:300])

    return all_docs

if __name__ == "__main__":
    docs = load_text("data/raw") # This now matches the function name!
    logger.info(f"Total documents loaded: {len(docs)}")