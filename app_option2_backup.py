import json
import re
from pathlib import Path

import streamlit as st
import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

CHUNKS_FILE = Path("data/chunks/chunks.json")
CHROMA_DIR = "data/chroma_bge"
COLLECTION_NAME = "papers_bge"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "openai/gpt-oss-20b"

PAPER_TITLES = {
    "01_plate_tectonic_marine_diversity.pdf":
        "Plate tectonic regulation of global marine animal diversity",
    "02_global_biogeography_pangaea.pdf":
        "Global biogeography of terrestrial vertebrates after Pangaea",
    "03_pangaea_terrestrial_biodiversity.pdf":
        "Pangaea fragmentation and terrestrial biodiversity",
    "04_supercontinent_cycle_climate.pdf":
        "Supercontinent cycle and long-term climate",
    "05_cambrian_geography_climate.pdf":
        "Palaeogeographic configuration and early Cambrian climate",
    "06_phanerozoic_landscape_biosphere.pdf":
        "Phanerozoic landscape and biosphere",
    "07_tropical_forest_biodiversity.pdf":
        "Tropical forest biodiversity",
    "08_zealandia_terrestrial_fauna.pdf":
        "Terrestrial fauna of Zealandia",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_rag_resources():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedding_model = TextEmbedding(MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    tokenized_corpus = [
        re.findall(r"\b\w+\b", chunk["text"].lower())
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    chunk_lookup = {
        (
            chunk["source"],
            chunk["page"],
            chunk["chunk"],
        ): chunk
        for chunk in chunks
    }

    return (
        chunks,
        embedding_model,
        collection,
        bm25,
        chunk_lookup,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def hybrid_retrieve(
    question,
    chunks,
    embedding_model,
    collection,
    bm25,
    chunk_lookup,
    top_k=3,
):

    # -------------------------
    # Dense retrieval
    # -------------------------

    query_embedding = list(
        embedding_model.embed([question])
    )[0].tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
    )

    dense_items = []

    for i in range(len(dense_results["ids"][0])):

        metadata = dense_results["metadatas"][0][i]

        key = (
            metadata["source"],
            int(metadata["page"]),
            int(metadata["chunk"]),
        )

        dense_items.append(key)


    # -------------------------
    # BM25 retrieval
    # -------------------------

    tokenized_query = re.findall(r"\b\w+\b", question.lower())

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:10]

    bm25_items = []

    for index in bm25_indices:

        chunk = chunks[index]

        key = (
            chunk["source"],
            int(chunk["page"]),
            int(chunk["chunk"]),
        )

        bm25_items.append(key)


    # -------------------------
    # Reciprocal Rank Fusion
    # -------------------------

    scores = {}

    for rank, key in enumerate(dense_items, start=1):
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)

    for rank, key in enumerate(bm25_items, start=1):
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)


    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:top_k]


    results = []

    for key in ranked_keys:

        chunk = chunk_lookup[key]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk": chunk["chunk"],
            "title": PAPER_TITLES.get(
                chunk["source"],
                chunk["source"],
            ),
            "rrf": scores[key],
        })

    return results


# ============================================================
# GROQ ANSWER
# ============================================================

def generate_answer(question, results):

    import os

    # Prefer the environment variable used by the terminal.
    api_key = os.environ.get("GROQ_API_KEY")

    # Optionally fall back to Streamlit secrets.
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    client = Groq(api_key=api_key)

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
[Source {i}]
Paper: {result['title']}
File: {result['source']}
Page: {result['page']}
Chunk: {result['chunk']}

Passage:
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are a research-paper question answering assistant.

Answer ONLY using the supplied research-paper passages.

Do not use outside knowledge.

If the passages do not contain enough information to answer the
question, say exactly:

"I don't know based on the provided papers."

When making claims, cite the supporting passage using [1], [2],
or [3].

Keep the answer clear and concise.

Do not invent citations.
"""

    user_prompt = f"""
Question:
{question}

Retrieved research-paper passages:
{context}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_completion_tokens=700,
        reasoning_effort="low",
    )

    return response.choices[0].message.content


# ============================================================
# UI
# ============================================================

st.title("📚 Research Paper RAG")
st.caption(
    "Grounded question answering over 8 research papers "
    "using BGE + BM25 hybrid retrieval + Groq."
)

st.divider()

question = st.text_area(
    "Ask a question about the research papers",
    placeholder=(
        "Example: What mechanisms connect continental "
        "fragmentation with increased marine biodiversity?"
    ),
    height=100,
)

ask = st.button(
    "🔎 Search & Generate Answer",
    type="primary",
)


# ============================================================
# RUN
# ============================================================

if ask:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Loading RAG system..."):

            (
                chunks,
                embedding_model,
                collection,
                bm25,
                chunk_lookup,
            ) = load_rag_resources()

        with st.spinner("Retrieving supporting passages..."):

            results = hybrid_retrieve(
                question,
                chunks,
                embedding_model,
                collection,
                bm25,
                chunk_lookup,
                top_k=3,
            )

        with st.spinner("Generating grounded answer..."):

            try:
                answer = generate_answer(
                    question,
                    results,
                )

            except Exception as e:

                st.error(
                    f"Could not generate the answer: {e}"
                )
                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("🤖 RAG Answer")

        st.write(answer)


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader("📚 Sources")

        for i, result in enumerate(results, start=1):

            st.markdown(
                f"**[{i}] {result['title']}**  \n"
                f"Page **{result['page']}** · "
                f"Chunk **{result['chunk']}**"
            )


        # ====================================================
        # PASSAGES
        # ====================================================

        st.subheader("📖 Top 3 Supporting Passages")

        for i, result in enumerate(results, start=1):

            with st.expander(
                f"[{i}] {result['title']} — Page {result['page']}"
            ):

                st.markdown(
                    f"**File:** `{result['source']}`"
                )

                st.markdown(
                    f"**Page:** {result['page']}"
                )

                st.markdown(
                    f"**Chunk:** {result['chunk']}"
                )

                st.markdown(
                    f"**RRF score:** {result['rrf']:.6f}"
                )

                st.write(result["text"])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG System")

    st.write(
        "This application uses:"
    )

    st.markdown(
        """
- **BGE-small** embeddings
- **ChromaDB** vector database
- **BM25** keyword retrieval
- **RRF** hybrid retrieval
- **Groq GPT-OSS-20B**
- **Top-3 supporting passages**
- **Paper/page citations**
        """
    )

    st.divider()

    st.caption(
        "Analytics Vidhya Pinnacle Plus Capstone"
    )
