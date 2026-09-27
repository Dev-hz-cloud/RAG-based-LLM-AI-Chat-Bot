import json
import os
import re
from pathlib import Path

import chromadb
import streamlit as st
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from groq import Groq
from ddgs import DDGS


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "data" / "chroma_bge"
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "chunks.json"

COLLECTION_NAME = "papers_bge"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K_DENSE = 10
TOP_K_BM25 = 10
FINAL_TOP_K = 3
RRF_K = 60

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Paper RAG")
st.caption(
    "Hybrid Retrieval • Conversational Memory • Paper + Web Search"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_system():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    embedding_model = TextEmbedding(
        model_name=EMBED_MODEL
    )

    tokenized_chunks = [
        re.findall(
            r"\b\w+\b",
            chunk["text"].lower()
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return (
        chunks,
        collection,
        embedding_model,
        bm25,
    )


chunks, collection, embedding_model, bm25 = load_system()


# ============================================================
# GROQ
# ============================================================

def get_groq():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_web_results" not in st.session_state:
    st.session_state.last_web_results = []

if "last_mode" not in st.session_state:
    st.session_state.last_mode = ""


# ============================================================
# RETRIEVAL — LOCAL PAPERS
# ============================================================

def retrieve(question):

    # -----------------------------
    # Dense retrieval
    # -----------------------------

    query_embedding = list(
        embedding_model.embed([question])
    )[0].tolist()

    dense_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_DENSE,
    )

    dense_ids = dense_result["ids"][0]

    dense_indices = []

    for chunk_id in dense_ids:

        try:
            dense_indices.append(
                int(str(chunk_id).split("_")[-1])
            )
        except ValueError:
            continue


    # -----------------------------
    # BM25 retrieval
    # -----------------------------

    query_tokens = re.findall(
        r"\b\w+\b",
        question.lower()
    )

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:TOP_K_BM25]


    # -----------------------------
    # Reciprocal Rank Fusion
    # -----------------------------

    scores = {}

    for rank, idx in enumerate(dense_indices):

        scores[idx] = scores.get(idx, 0) + (
            1 / (RRF_K + rank + 1)
        )

    for rank, idx in enumerate(bm25_indices):

        scores[idx] = scores.get(idx, 0) + (
            1 / (RRF_K + rank + 1)
        )


    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for idx, score in ranked[:FINAL_TOP_K]:

        if 0 <= idx < len(chunks):

            result = dict(chunks[idx])
            result["rrf_score"] = score
            results.append(result)

    return results


# ============================================================
# WEB SEARCH — DIRECT
# ============================================================

def web_search(question):

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                question,
                max_results=5,
            )

            for item in search_results:

                results.append({
                    "title": item.get(
                        "title",
                        "Untitled",
                    ),
                    "url": item.get(
                        "href",
                        "",
                    ),
                    "snippet": item.get(
                        "body",
                        "",
                    ),
                })

    except Exception as e:

        st.error(
            f"Web search error: {e}"
        )

    return results


# ============================================================
# QUESTION REWRITE FOR CHAT MEMORY
# ============================================================

def rewrite_question(question):

    if not st.session_state.chat_history:
        return question

    groq = get_groq()

    if groq is None:
        return question

    recent_history = st.session_state.chat_history[-6:]

    history_text = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in recent_history
    )

    prompt = f"""
Rewrite the user's latest question into a standalone
search query using the conversation context.

Do not add facts that are not present in the conversation.
Keep the original meaning.
If the question is already standalone, return it unchanged.

Conversation:
{history_text}

Latest question:
{question}

Return ONLY the rewritten search query.
"""

    try:

        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
        )

        rewritten = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        return rewritten or question

    except Exception:
        return question


# ============================================================
# PAPER ANSWER
# ============================================================

def generate_paper_answer(
    question,
    results,
):

    groq = get_groq()

    if groq is None:
        return (
            "GROQ_API_KEY is not set. "
            "Run: read -s GROQ_API_KEY; export GROQ_API_KEY"
        )

    context_parts = []

    for i, result in enumerate(results, 1):

        context_parts.append(
            f"""
[P{i}]
Paper source: {result.get("source", "Unknown")}
Page: {result.get("page", "Unknown")}

{result.get("text", "")}
"""
        )

    context = "\n".join(context_parts)

    history = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in st.session_state.chat_history[-6:]
    )

    prompt = f"""
You are answering questions using a collection of research papers.

IMPORTANT:
- Use ONLY the supplied research-paper passages as factual evidence.
- Do not use outside knowledge.
- If the passages do not contain enough information, say:
  "I don't know based on the provided papers."
- Do not invent citations.
- Cite supporting passages using [P1], [P2], or [P3].
- Be concise but explanatory.
- Conversation history is only for resolving references such as "it",
  "that", or "the previous paper".

Conversation history:
{history}

Current question:
{question}

Retrieved research-paper passages:
{context}

Answer:
"""

    try:

        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
        )

        return (
            response.choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        return f"LLM error: {e}"


# ============================================================
# WEB ANSWER
# ============================================================

def generate_web_answer(
    question,
    web_results,
):

    groq = get_groq()

    if groq is None:
        return (
            "GROQ_API_KEY is not set. "
            "Run: read -s GROQ_API_KEY; export GROQ_API_KEY"
        )

    sources = []

    for i, result in enumerate(
        web_results,
        1,
    ):

        sources.append(
            f"""
[W{i}]
Title: {result["title"]}
URL: {result["url"]}
Content: {result["snippet"]}
"""
        )

    web_context = "\n".join(sources)

    history = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in st.session_state.chat_history[-6:]
    )

    prompt = f"""
You are answering a question using web-search results.

IMPORTANT:
- Use the supplied web results as your evidence.
- Do not use the local research-paper database.
- Do not invent information.
- Cite claims using [W1], [W2], etc.
- If the search results do not provide enough evidence, clearly say so.
- Conversation history is only for resolving references.

Conversation history:
{history}

Current question:
{question}

Web-search results:
{web_context}

Answer:
"""

    try:

        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
        )

        return (
            response.choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        return f"LLM error: {e}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Controls")

    st.write(
        f"**Messages in memory:** "
        f"{len(st.session_state.chat_history)}"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.chat_history = []
        st.session_state.last_results = []
        st.session_state.last_web_results = []
        st.session_state.last_mode = ""

        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# MODE SELECTOR
# ============================================================

mode = st.radio(
    "Search mechanism",
    [
        "📚 Research Papers",
        "🌐 Web Search",
    ],
    horizontal=True,
)

if mode == "📚 Research Papers":

    st.caption(
        "Uses BGE + BM25 + RRF over the local research papers."
    )

else:

    st.caption(
        "Always performs a direct web search using DDGS."
    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question..."
)


if question:

    # Display user message immediately.
    with st.chat_message("user"):
        st.markdown(question)

    previous_history = list(
        st.session_state.chat_history
    )

    # Save user message.
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # Resolve conversational references.
    search_question = rewrite_question(
        question
    )


    # ========================================================
    # RESEARCH PAPER MODE
    # ========================================================

    if mode == "📚 Research Papers":

        results = retrieve(
            search_question
        )

        answer = generate_paper_answer(
            question,
            results,
        )

        st.session_state.last_results = results
        st.session_state.last_web_results = []
        st.session_state.last_mode = (
            "📚 Research Papers → Local RAG"
        )


    # ========================================================
    # WEB SEARCH MODE
    # ========================================================

    else:

        # IMPORTANT:
        # Web mode NEVER retrieves local papers.
        # Web mode NEVER checks paper sufficiency.
        # Web mode NEVER falls back from papers.
        # Selecting Web Search directly triggers DDGS.

        web_results = web_search(
            search_question
        )

        answer = generate_web_answer(
            question,
            web_results,
        )

        st.session_state.last_results = []
        st.session_state.last_web_results = (
            web_results
        )
        st.session_state.last_mode = (
            "🌐 Web Search → Direct web search"
        )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
    })

    st.rerun()


# ============================================================
# LATEST ANSWER SOURCES
# ============================================================

if st.session_state.last_mode:

    st.divider()

    st.caption(
        f"**Mode:** {st.session_state.last_mode}"
    )


# ============================================================
# PAPER SOURCES
# ============================================================

if (
    st.session_state.last_results
    and st.session_state.last_mode.startswith(
        "📚"
    )
):

    st.subheader(
        "📖 Top 3 Supporting Passages"
    )

    for i, result in enumerate(
        st.session_state.last_results,
        1,
    ):

        source = result.get(
            "source",
            "Unknown",
        )

        page = result.get(
            "page",
            "Unknown",
        )

        score = result.get(
            "rrf_score",
            0,
        )

        with st.expander(
            f"[P{i}] {source} — Page {page}"
        ):

            st.caption(
                f"RRF score: {score:.6f}"
            )

            st.write(
                result.get(
                    "text",
                    "",
                )
            )


# ============================================================
# WEB SOURCES
# ============================================================

if (
    st.session_state.last_web_results
    and st.session_state.last_mode.startswith(
        "🌐"
    )
):

    st.subheader(
        "🌐 Web Search Results"
    )

    for i, result in enumerate(
        st.session_state.last_web_results,
        1,
    ):

        with st.expander(
            f"[W{i}] {result['title']}"
        ):

            if result["url"]:
                st.markdown(
                    f"**Source:** {result['url']}"
                )

            st.write(
                result["snippet"]
            )
