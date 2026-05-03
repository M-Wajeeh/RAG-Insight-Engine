import os

# Folders (TOP LEVEL — not inside data)
folders = [
    "data/raw",
    "data/processed",

    "loaders",
    "splitters",
    "embeddings",
    "vectorstore",
    "retrievers",
    "llm",
    "pipeline",
    "utils"
]

files = [
    "loaders/pdf_loader.py",
    "loaders/text_loader.py",

    "splitters/chunker.py",

    "embeddings/embedder.py",

    "vectorstore/chroma_store.py",

    "retrievers/base_retriever.py",
    "retrievers/mmr_retriever.py",

    "llm/generator.py",

    "pipeline/rag_pipeline.py",

    "utils/logger.py",
    "utils/config.py",

    "app.py",
    "requirements.txt",
    "README.md"
]

def create_structure():
    print(" Creating your RAG structure...\n")

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f" Created: {folder}")

        # Add __init__.py (except data folders)
        if not folder.startswith("data"):
            init_path = os.path.join(folder, "__init__.py")
            with open(init_path, "w") as f:
                pass
            print(f"   └── __init__.py")

    # Create files
    for file in files:
        # Ensure parent dirs exist (safety); dirname is "" for top-level files
        parent_dir = os.path.dirname(file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(file, "w") as f:
            pass

        print(f" Created: {file}")

    print("\n Done. Your structure is ready.")


if __name__ == "__main__":
    create_structure()
