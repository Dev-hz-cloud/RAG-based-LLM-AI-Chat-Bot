from pathlib import Path
import json
import chromadb
from fastembed import TextEmbedding

CHUNKS_FILE = Path("data/chunks/chunks.json")
DB_DIR = "data/chroma_minilm"
COLLECTION_NAME = "papers_minilm"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks):,} chunks.")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = TextEmbedding(model_name=MODEL_NAME)
    print("Embedding model ready.")

    print("\nStarting ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk": chunk["chunk"],
        })

    print("\nCreating MiniLM embeddings...")
    embeddings = list(model.embed(documents))
    print(f"Created {len(embeddings):,} embeddings.")

    print("\nAdding data to ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[embedding.tolist() for embedding in embeddings],
    )

    print("\n✅ MiniLM indexing complete!")
    print(f"Documents indexed: {collection.count():,}")
    print(f"Database location: {DB_DIR}")

if __name__ == "__main__":
    main()
