import json
import re
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from groq import Groq


CHUNKS_FILE = Path("data/chunks/chunks.json")

BGE_DB_DIR = "data/chroma_bge"
BGE_COLLECTION = "papers_bge"
BGE_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K_DENSE = 10
TOP_K_BM25 = 10
FINAL_TOP_K = 3
RRF_K = 60

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


print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks):,} chunks.")


chunk_lookup = {
    (chunk["source"], chunk["page"], chunk["chunk"]): i
    for i, chunk in enumerate(chunks)
}


print("\nLoading BGE embedding model...")

embedding_model = TextEmbedding(
    model_name=BGE_MODEL
)

print("BGE model ready.")


print("\nOpening BGE ChromaDB...")

client = chromadb.PersistentClient(
    path=BGE_DB_DIR
)

collection = client.get_collection(
    BGE_COLLECTION
)

print(
    f"ChromaDB contains {collection.count():,} chunks."
)


print("\nBuilding BM25 index...")

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 ready.")


print("\nConnecting to Groq...")

groq_client = Groq()

print("Groq ready.")


def reciprocal_rank_fusion(
    dense_results,
    bm25_results
):
    scores = {}

    for rank, chunk_index in enumerate(dense_results):
        scores[chunk_index] = (
            scores.get(chunk_index, 0)
            + 1 / (RRF_K + rank + 1)
        )

    for rank, chunk_index in enumerate(bm25_results):
        scores[chunk_index] = (
            scores.get(chunk_index, 0)
            + 1 / (RRF_K + rank + 1)
        )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def retrieve(question):

    query_embedding = list(
        embedding_model.embed([question])
    )[0]

    dense_results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K_DENSE
    )

    dense_indices = []

    for metadata in dense_results["metadatas"][0]:

        key = (
            metadata["source"],
            metadata["page"],
            metadata["chunk"]
        )

        if key in chunk_lookup:
            dense_indices.append(
                chunk_lookup[key]
            )


    tokenized_question = re.findall(r"\b\w+\b", question.lower())

    bm25_scores = bm25.get_scores(
        tokenized_question
    )

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:TOP_K_BM25]


    fused_results = reciprocal_rank_fusion(
        dense_indices,
        bm25_indices
    )


    final_results = []

    for chunk_index, score in fused_results[:FINAL_TOP_K]:

        chunk = chunks[chunk_index]
        source = chunk["source"]

        final_results.append({
            "text": chunk["text"],
            "source": source,
            "title": PAPER_TITLES.get(
                source,
                source
            ),
            "page": chunk["page"],
            "chunk": chunk["chunk"],
            "rrf_score": score
        })

    return final_results


def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
[Passage {i}]
Paper: {result["title"]}
Page: {result["page"]}

{result["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):

    context = build_context(results)

    system_prompt = """
You are a scientific research assistant.

Answer the user's question using ONLY
the information contained in the provided
research-paper passages.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. If the passages do not contain enough
   information to answer the question,
   say exactly:
   "I don't know based on the provided papers."

4. Give a clear and concise answer.

5. Cite supporting passages using [1],
   [2], or [3].

6. Only cite a passage when it actually
   supports the statement.

7. Do not mention information that is
   absent from the retrieved passages.
"""

    user_prompt = f"""
Question:
{question}

Retrieved passages:
{context}

Answer the question using only the
retrieved passages.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_completion_tokens=700,
        reasoning_effort="low"
    )

    return response.choices[0].message.content


def display_results(question, results, answer):

    print("\n")
    print("=" * 80)
    print("RAG ANSWER")
    print("=" * 80)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)


    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(
            f"[{i}] "
            f"{result['title']} "
            f"— Page {result['page']}"
        )


    print("\n")
    print("=" * 80)
    print("TOP 3 SUPPORTING PASSAGES")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\n--- Source {i} ---")

        print(f"Paper : {result['title']}")
        print(f"File  : {result['source']}")
        print(f"Page  : {result['page']}")
        print(f"Chunk : {result['chunk']}")
        print(f"RRF   : {result['rrf_score']:.6f}")

        print("\nPassage:")
        print(result["text"][:700])

    print("\n" + "=" * 80)


def main():

    question = input(
        "\nAsk a question about the research papers: "
    ).strip()

    if not question:
        print("No question provided.")
        return

    print("\nRetrieving supporting passages...")

    results = retrieve(question)

    print("Generating grounded answer with Groq...")

    answer = generate_answer(
        question,
        results
    )

    display_results(
        question,
        results,
        answer
    )


if __name__ == "__main__":
    main()
