from pathlib import Path
import json
import chromadb
from fastembed import TextEmbedding

QUESTIONS_FILE = Path("data/evaluation_questions.json")
DB_DIR = "data/chroma_minilm"
COLLECTION_NAME = "papers_minilm"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_FILE = Path("data/evaluation_results_minilm.json")


def main():
    print("Loading evaluation questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    print("\nLoading MiniLM...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening MiniLM ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = []
    hits = 0

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = item["expected_sources"]

        print(f"\n[{i}/{len(questions)}] {question}")

        query_embedding = list(model.embed([question]))[0]

        search_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
        )

        top_results = []

        for rank in range(3):
            metadata = search_results["metadatas"][0][rank]
            distance = search_results["distances"][0][rank]

            top_results.append({
                "rank": rank + 1,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "distance": distance,
            })

        hit_at_3 = any(
            result["source"] in expected_sources
            for result in top_results
        )

        if hit_at_3:
            hits += 1

        results.append({
            "question_number": i,
            "question": question,
            "expected_sources": expected_sources,
            "top_3": top_results,
            "hit_at_3": hit_at_3,
        })

        print(f"Expected: {expected_sources}")
        print(f"Hit@3: {'YES' if hit_at_3 else 'NO'}")

    hit_rate = hits / len(questions) * 100

    output = {
        "model": MODEL_NAME,
        "database": DB_DIR,
        "total_questions": len(questions),
        "hits": hits,
        "hit_at_3_percent": hit_rate,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("MINILM EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Hits@3: {hits}/{len(questions)}")
    print(f"Hit@3:  {hit_rate:.1f}%")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
