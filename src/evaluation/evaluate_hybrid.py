from pathlib import Path
import json
import re
import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

QUESTIONS_FILE = Path("data/evaluation_questions.json")
CHUNKS_FILE = Path("data/chunks/chunks.json")
OUTPUT_FILE = Path("data/evaluation_results_hybrid.json")

DB_DIR = "data/chroma_bge"
COLLECTION_NAME = "papers_bge"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

RRF_K = 60
CANDIDATE_K = 10


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks):,} chunks.")

    print("\nBuilding BM25 index...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index ready.")

    print("\nLoading BGE embedding model...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print("\nLoading evaluation questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    results = []

    print("\nRunning hybrid evaluation...")

    for item in questions:
        question = item["question"]
        expected_sources = set(item["expected_sources"])

        # -----------------------------
        # Dense retrieval
        # -----------------------------
        query_embedding = list(model.embed([question]))[0]

        dense_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=CANDIDATE_K,
        )

        dense_ids = []

        for metadata in dense_results["metadatas"][0]:
            source = metadata["source"]
            page = metadata["page"]
            chunk_number = metadata["chunk"]

            # Find the corresponding chunk index.
            for index, chunk in enumerate(chunks):
                if (
                    chunk["source"] == source
                    and chunk["page"] == page
                    and chunk["chunk"] == chunk_number
                ):
                    dense_ids.append(index)
                    break

        # -----------------------------
        # BM25 retrieval
        # -----------------------------
        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)
        bm25_indices = scores.argsort()[::-1][:CANDIDATE_K].tolist()

        # -----------------------------
        # Reciprocal Rank Fusion
        # -----------------------------
        rrf_scores = {}

        for rank, index in enumerate(dense_ids, start=1):
            rrf_scores[index] = rrf_scores.get(index, 0) + (
                1 / (RRF_K + rank)
            )

        for rank, index in enumerate(bm25_indices, start=1):
            rrf_scores[index] = rrf_scores.get(index, 0) + (
                1 / (RRF_K + rank)
            )

        top_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:3]

        retrieved = []

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[index]

            retrieved.append({
                "rank": rank,
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "rrf_score": round(float(rrf_scores[index]), 6),
            })

        retrieved_sources = {
            result["source"] for result in retrieved
        }

        hit_at_3 = bool(expected_sources & retrieved_sources)

        results.append({
            "id": item["id"],
            "question": question,
            "expected_sources": item["expected_sources"],
            "retrieved_results": retrieved,
            "hit_at_3": hit_at_3,
        })

        status = "HIT" if hit_at_3 else "MISS"
        print(f"  Q{item['id']:02d}: {status}")

    hits = sum(result["hit_at_3"] for result in results)
    total = len(results)

    evaluation = {
        "retrieval_strategy": "Hybrid Retrieval (Dense + BM25 + RRF)",
        "dense_model": MODEL_NAME,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "top_k": 3,
        "results": results,
        "summary": {
            "total_questions": total,
            "hits_at_3": hits,
            "misses_at_3": total - hits,
            "hit_at_3": round(hits / total, 4) if total else 0,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Questions : {total}")
    print(f"Hits @3   : {hits}")
    print(f"Misses @3 : {total - hits}")
    print(f"Hit@3     : {hits / total:.1%}" if total else "Hit@3     : 0%")
    print(f"\nSaved complete results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
