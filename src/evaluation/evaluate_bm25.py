from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi

QUESTIONS_FILE = Path("data/evaluation_questions.json")
CHUNKS_FILE = Path("data/chunks/chunks.json")
OUTPUT_FILE = Path("data/evaluation_results_bm25.json")


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

    print("\nLoading evaluation questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    results = []

    print("\nRunning evaluation...")

    for item in questions:
        question = item["question"]
        expected_sources = set(item["expected_sources"])

        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[::-1][:3]

        retrieved = []

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[index]

            retrieved.append({
                "rank": rank,
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "score": round(float(scores[index]), 6),
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
        "retrieval_strategy": "BM25 Keyword Retrieval",
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
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Questions : {total}")
    print(f"Hits @3   : {hits}")
    print(f"Misses @3 : {total - hits}")
    print(f"Hit@3     : {hits / total:.1%}" if total else "Hit@3     : 0%")
    print(f"\nSaved complete results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
