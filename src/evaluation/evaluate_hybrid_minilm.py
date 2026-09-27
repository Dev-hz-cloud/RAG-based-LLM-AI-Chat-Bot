from pathlib import Path
import json
import re

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/chunks/chunks.json")
QUESTIONS_FILE = Path("data/evaluation_questions.json")

DB_DIR = "data/chroma_minilm"
COLLECTION_NAME = "papers_minilm"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_FILE = Path("data/evaluation_results_hybrid_minilm.json")

RRF_K = 60


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

    print("\nLoading MiniLM...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening MiniLM ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print("\nLoading evaluation questions...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    results = []
    hits = 0

    for q_num, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = item["expected_sources"]

        print(f"\n[{q_num}/{len(questions)}] {question}")

        # -------------------------
        # MiniLM dense retrieval
        # -------------------------
        query_embedding = list(model.embed([question]))[0]

        dense_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10,
        )

        dense_ranked = []

        for rank in range(10):
            metadata = dense_results["metadatas"][0][rank]

            dense_ranked.append({
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
            })

        # -------------------------
        # BM25 retrieval
        # -------------------------
        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)

        bm25_indices = scores.argsort()[::-1][:10]

        bm25_ranked = []

        for index in bm25_indices:
            bm25_ranked.append({
                "source": chunks[index]["source"],
                "page": chunks[index]["page"],
                "chunk": chunks[index]["chunk"],
            })

        # -------------------------
        # Reciprocal Rank Fusion
        # -------------------------
        rrf_scores = {}

        for rank, result in enumerate(dense_ranked, start=1):
            key = (
                result["source"],
                result["page"],
                result["chunk"],
            )

            rrf_scores.setdefault(key, 0)
            rrf_scores[key] += 1 / (RRF_K + rank)

        for rank, result in enumerate(bm25_ranked, start=1):
            key = (
                result["source"],
                result["page"],
                result["chunk"],
            )

            rrf_scores.setdefault(key, 0)
            rrf_scores[key] += 1 / (RRF_K + rank)

        ranked_hybrid = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        top_results = []

        for rank, (key, score) in enumerate(ranked_hybrid, start=1):
            source, page, chunk = key

            top_results.append({
                "rank": rank,
                "source": source,
                "page": page,
                "chunk": chunk,
                "rrf_score": score,
            })

        hit_at_3 = any(
            result["source"] in expected_sources
            for result in top_results
        )

        if hit_at_3:
            hits += 1

        results.append({
            "question_number": q_num,
            "question": question,
            "expected_sources": expected_sources,
            "top_3": top_results,
            "hit_at_3": hit_at_3,
        })

        print(f"Expected: {expected_sources}")
        print(f"Hit@3: {'YES' if hit_at_3 else 'NO'}")

    hit_rate = hits / len(questions) * 100

    output = {
        "retrieval_method": "MiniLM + BM25 + RRF",
        "embedding_model": MODEL_NAME,
        "total_questions": len(questions),
        "hits": hits,
        "hit_at_3_percent": hit_rate,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("MINILM + BM25 HYBRID EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Hits@3: {hits}/{len(questions)}")
    print(f"Hit@3:  {hit_rate:.1f}%")
    print(f"Saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
