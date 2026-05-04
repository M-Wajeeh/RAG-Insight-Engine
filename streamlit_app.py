import streamlit as st

from pipeline.rag_pipeline import RAGPipeline
from utils.config import DATA_PATH, DEFAULT_RETRIEVER, TOP_K


st.set_page_config(page_title="RAG Insight Engine", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Initializing RAG pipeline...")
def get_pipeline(retriever_kind: str, top_k: int) -> RAGPipeline:
    """Cache one pipeline per (retriever_kind, top_k) so embeddings/DB are reused."""
    return RAGPipeline(
        retriever_kind=retriever_kind,
        top_k=top_k,
        data_path=DATA_PATH,
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources retrieved.")
        return

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled"
        file_name = source.get("file_name") or "unknown"
        page = source.get("page")
        page_str = f"page {page}" if page not in (None, "") else "page N/A"
        st.markdown(f"**[{index}]** `{file_name}` · {page_str} · _{title}_")


def main() -> None:
    st.title("RAG Insight Engine")
    st.caption("Ask questions grounded in the documents under `data/raw`.")

    with st.sidebar:
        st.header("Settings")
        retriever_kind = st.selectbox(
            "Retriever",
            options=["similarity", "mmr"],
            index=0 if DEFAULT_RETRIEVER == "similarity" else 1,
            help="`mmr` adds diversity to retrieved chunks.",
        )
        top_k = st.slider("Top K chunks", min_value=1, max_value=20, value=TOP_K)
        st.caption("Pipeline rebuilds only when these settings change.")

    pipeline = get_pipeline(retriever_kind, top_k)

    query = st.text_input(
        "Question",
        placeholder="e.g. What is the main contribution of the paper?",
    )
    submitted = st.button("Ask", type="primary", disabled=not query.strip())

    if submitted:
        with st.spinner("Retrieving and generating..."):
            result = pipeline.ask(query)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        render_sources(result["sources"])

        with st.expander("Raw structured output"):
            st.json(result)


if __name__ == "__main__":
    main()
