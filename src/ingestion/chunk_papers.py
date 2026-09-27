from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.load_papers import load_all_papers


OUTPUT_DIR = Path("data/chunks")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


def main():
    print("Loading papers...")

    pages = load_all_papers()

    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk": chunk_number,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nCreated {len(chunks):,} chunks.")
    print(f"Saved to: {OUTPUT_FILE}")

    if chunks:
        print("\nFirst chunk:")
        print(f"  Source: {chunks[0]['source']}")
        print(f"  Page: {chunks[0]['page']}")
        print(f"  Chunk: {chunks[0]['chunk']}")
        print(f"  Characters: {len(chunks[0]['text'])}")
        print(f"  Text: {chunks[0]['text'][:300]}...")


if __name__ == "__main__":
    main()
