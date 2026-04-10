"""
Build persistent Chroma collections for both corpora.
Run once (or re-run to rebuild):
    python -m person_b.build_index
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from tqdm import tqdm

from shared.config import (
    FLAT_CORPUS_PATH, HIERARCHY_CORPUS_PATH,
    CHROMA_PATH, FLAT_COLLECTION, HIERARCHY_COLLECTION,
)
from shared.utils  import load_jsonl, batched
from person_b.embedder import embed


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def index_corpus(
    jsonl_path: Path,
    collection_name: str,
    batch_size: int = 128,
    rebuild: bool = False,
) -> None:
    """
    Upsert all chunks from a .jsonl file into a named Chroma collection.
    Set rebuild=True to delete the collection and start fresh.
    """
    client = _get_client()

    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"[build_index] deleted existing collection '{collection_name}'")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = load_jsonl(jsonl_path)
    print(f"[build_index] indexing {len(chunks)} chunks → '{collection_name}'")

    for batch in tqdm(list(batched(chunks, batch_size)), desc=collection_name):
        ids        = [c["chunk_id"] for c in batch]
        texts      = [c["text"]     for c in batch]
        # Chroma metadata values must be str | int | float | bool
        metadatas  = [
            {
                "source":     c.get("source", ""),
                "section":    c.get("section", ""),
                "breadcrumb": c.get("breadcrumb", ""),
                "chunk_type": c.get("chunk_type", ""),
                "heading":    c.get("metadata", {}).get("heading", ""),
            }
            for c in batch
        ]
        embeddings = embed(texts)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"[build_index] ✓ '{collection_name}' now has {col.count()} docs")


def main(rebuild: bool = False):
    if not FLAT_CORPUS_PATH.exists():
        print(f"[ERROR] {FLAT_CORPUS_PATH} not found. Run person_a first.")
        return
    if not HIERARCHY_CORPUS_PATH.exists():
        print(f"[ERROR] {HIERARCHY_CORPUS_PATH} not found. Run person_a first.")
        return

    index_corpus(FLAT_CORPUS_PATH,      FLAT_COLLECTION,      rebuild=rebuild)
    index_corpus(HIERARCHY_CORPUS_PATH, HIERARCHY_COLLECTION, rebuild=rebuild)
    print("\n✓ Person B indexing complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="Delete and recreate collections from scratch")
    args = p.parse_args()
    main(rebuild=args.rebuild)
