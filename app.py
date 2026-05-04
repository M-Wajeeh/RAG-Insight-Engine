import json

from pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    query = "Tell about Table 1"
    result = pipeline.ask(query)

    print("\n=== QUERY ===")
    print(query)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    print(json.dumps(result["sources"], indent=2, default=str))


if __name__ == "__main__":
    main()
