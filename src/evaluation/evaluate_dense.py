from pathlib import Path
import json
import chromadb
from fastembed import TextEmbedding

QUESTIONS_FILE = Path("data/evaluation_questions.json")
OUTPUT_FILE = Path("data/evaluation_results_dense.json")

DB_DIR = "data/chroma_bge"
COLLECTION_NAME = "papers_bge"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("Loading evaluation questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    print("\nLoading embedding model...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = []

    print("\nRunning evaluation...")

    for item in questions:
        question = item["question"]
        expected_sources = set(item["expected_sources"])

        query_embedding = list(model.embed([question]))[0]

        search_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
        )

        retrieved = []

        for rank in range(3):
            metadata = search_results["metadatas"][0][rank]

            retrieved.append({
                "rank": rank + 1,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "distance": round(
                    float(search_results["distances"][0][rank]), 6
                ),
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
        "model": MODEL_NAME,
        "retrieval_strategy": "Dense Retrieval (Cosine Similarity)",
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
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Questions : {total}")
    print(f"Hits @3   : {hits}")
    print(f"Misses @3 : {total - hits}")
    print(f"Hit@3     : {hits / total:.1%}" if total else "Hit@3     : 0%")
    print(f"\nSaved complete results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
