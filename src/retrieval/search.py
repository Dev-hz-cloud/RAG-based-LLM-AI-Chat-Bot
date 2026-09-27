from pathlib import Path
import chromadb
from fastembed import TextEmbedding

DB_DIR = "data/chroma_bge"
COLLECTION_NAME = "papers_bge"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("Loading embedding model...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    question = input("\nAsk a question about the papers: ")

    print("\nSearching...")
    query_embedding = list(model.embed([question]))[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
    )

    print("\n" + "=" * 70)
    print("TOP 3 SUPPORTING PASSAGES")
    print("=" * 70)

    for i in range(3):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        distance = results["distances"][0][i]

        print(f"\n--- Result {i + 1} ---")
        print(f"Source : {metadata['source']}")
        print(f"Page   : {metadata['page']}")
        print(f"Chunk  : {metadata['chunk']}")
        print(f"Distance: {distance:.4f}")
        print(f"\n{document[:700]}")


if __name__ == "__main__":
    main()
