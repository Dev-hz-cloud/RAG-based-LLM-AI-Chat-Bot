import json
from pathlib import Path
from rag.rag_pipeline import retrieve, generate_answer

QUESTIONS_FILE = Path("data/evaluation_questions.json")
OUTPUT_FILE = Path("data/evaluation_results_rag.json")

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

print("=" * 70)
print("RAG ANSWER EVALUATION")
print("=" * 70)

for item in questions:
    question = item["question"]
    expected = item["expected_sources"]

    print(f"\nQ{item['id']}: {question}")

    try:
        retrieved = retrieve(question)
        answer = generate_answer(question, retrieved)

        retrieved_sources = [r["source"] for r in retrieved]
        hit = any(src in retrieved_sources for src in expected)

        result = {
            "id": item["id"],
            "question": question,
            "expected_sources": expected,
            "retrieved_sources": retrieved_sources,
            "retrieved_pages": [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "title": r.get("title", ""),
                }
                for r in retrieved
            ],
            "answer": answer,
            "source_hit": hit,
        }

        results.append(result)

        print(f"Source hit: {'YES' if hit else 'NO'}")
        print(f"Answer: {answer[:300]}...")

    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "id": item["id"],
            "question": question,
            "error": str(e),
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

successful = [r for r in results if "source_hit" in r]
hits = sum(r["source_hit"] for r in successful)

print("\n" + "=" * 70)
print("FINAL RAG EVALUATION")
print("=" * 70)
print(f"Questions evaluated: {len(successful)}/{len(questions)}")
print(f"Source hits: {hits}/{len(successful)}")

if successful:
    print(f"Source Hit@3: {hits / len(successful) * 100:.1f}%")

print(f"\nSaved to: {OUTPUT_FILE}")
