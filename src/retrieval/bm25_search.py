from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/chunks/chunks.json")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks):,} chunks.")

    print("\nBuilding BM25 index...")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index ready.")

    question = input("\nAsk a question about the papers: ")

    query_tokens = tokenize(question)

    print("\nSearching...")

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:3]

    print("\n" + "=" * 70)
    print("TOP 3 BM25 SUPPORTING PASSAGES")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        print(f"\n--- Result {rank} ---")
        print(f"Source : {chunk['source']}")
        print(f"Page   : {chunk['page']}")
        print(f"Chunk  : {chunk['chunk']}")
        print(f"BM25 Score: {scores[index]:.4f}")

        print(f"\n{chunk['text'][:700]}")


if __name__ == "__main__":
    main()
