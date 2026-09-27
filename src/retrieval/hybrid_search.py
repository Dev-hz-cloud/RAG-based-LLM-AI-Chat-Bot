from pathlib import Path
import json
import re

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/chunks/chunks.json")
DB_DIR = "data/chroma_bge"
COLLECTION_NAME = "papers_bge"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    # -------------------------
    # Load chunks
    # -------------------------
    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks):,} chunks.")

    # -------------------------
    # Load BM25
    # -------------------------
    print("\nBuilding BM25 index...")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 ready.")

    # -------------------------
    # Load BGE + ChromaDB
    # -------------------------
    print("\nLoading BGE model...")

    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening ChromaDB...")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print("BGE + ChromaDB ready.")

    # -------------------------
    # Question
    # -------------------------
    question = input(
        "\nAsk a question about the papers: "
    )

    # -------------------------
    # BGE search
    # -------------------------
    print("\nRunning BGE dense search...")

    query_embedding = list(
        model.embed([question])
    )[0]

    dense_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
    )

    dense_ids = dense_results["ids"][0]

    # -------------------------
    # BM25 search
    # -------------------------
    print("Running BM25 search...")

    query_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = bm25_scores.argsort()[::-1][:10]

    bm25_ids = [
        f"chunk_{index}"
        for index in bm25_indices
    ]

    # -------------------------
    # Reciprocal Rank Fusion
    # -------------------------
    print("Combining rankings...")

    rrf_scores = {}

    k = 60

    for rank, chunk_id in enumerate(dense_ids, start=1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    top_hybrid = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:3]

    # -------------------------
    # Display results
    # -------------------------
    print("\n" + "=" * 70)
    print("TOP 3 HYBRID SUPPORTING PASSAGES")
    print("=" * 70)

    for rank, chunk_id in enumerate(top_hybrid, start=1):

        index = int(chunk_id.split("_")[1])
        chunk = chunks[index]

        print(f"\n--- Result {rank} ---")
        print(f"Source : {chunk['source']}")
        print(f"Page   : {chunk['page']}")
        print(f"Chunk  : {chunk['chunk']}")
        print(f"RRF Score: {rrf_scores[chunk_id]:.6f}")

        print(f"\n{chunk['text'][:700]}")


if __name__ == "__main__":
    main()
