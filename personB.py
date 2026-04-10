# ============================================================
# FILE: person_b/embedder.py
# ============================================================
"""
Wraps sentence-transformers so the rest of the codebase
never imports it directly (easy to swap to OpenAI later).
"""
from __future__ import annotations
from functools import lru_cache
from shared.config import EMBED_MODEL


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    print(f"[embedder] loading {EMBED_MODEL} …")
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of strings → list of float vectors.
    Handles batching internally for large inputs.
    """
    model = _load_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    return vecs.tolist()


# ============================================================
# FILE: person_b/build_index.py
# ============================================================
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


# ============================================================
# FILE: person_b/bm25_store.py
# ============================================================
"""
In-memory BM25 index built from a .jsonl corpus at startup.
Rebuilt each run (~5 s for 20k chunks — acceptable).
"""
from __future__ import annotations
from functools import lru_cache
from rank_bm25 import BM25Okapi
from shared.utils import load_jsonl


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, jsonl_path: str):
        self.chunks: list[dict] = load_jsonl(jsonl_path)
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.index = BM25Okapi(tokenized)
        print(f"[bm25] built index for {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 20) -> list[dict]:
        tokens = _tokenize(query)
        scores = self.index.get_scores(tokens)
        top_k  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in top_k:
            c = self.chunks[i].copy()
            c["bm25_score"] = float(scores[i])
            results.append(c)
        return results


# Lazy singletons — built once per process
@lru_cache(maxsize=1)
def get_flat_bm25() -> BM25Store:
    from shared.config import FLAT_CORPUS_PATH
    return BM25Store(str(FLAT_CORPUS_PATH))

@lru_cache(maxsize=1)
def get_hierarchy_bm25() -> BM25Store:
    from shared.config import HIERARCHY_CORPUS_PATH
    return BM25Store(str(HIERARCHY_CORPUS_PATH))


# ============================================================
# FILE: person_b/retrieve.py
# ============================================================
"""
The single public interface Person C and Person D import.

Usage:
    from person_b.retrieve import retrieve

    chunks = retrieve("What cooperative organizations are exempt?",
                      mode="hybrid", corpus="hierarchy", k=5)
"""
from __future__ import annotations
import chromadb
from functools import lru_cache

from shared.config import (
    CHROMA_PATH, FLAT_COLLECTION, HIERARCHY_COLLECTION, RETRIEVAL_K
)
from person_b.embedder  import embed
from person_b.bm25_store import get_flat_bm25, get_hierarchy_bm25


# ── Chroma helpers ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)

def _collection(corpus: str):
    name = FLAT_COLLECTION if corpus == "flat" else HIERARCHY_COLLECTION
    return _client().get_collection(name)


def _format_chroma_results(results: dict) -> list[dict]:
    """Convert Chroma's nested response dict into a flat list of chunk dicts."""
    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    ids    = results["ids"][0]
    dists  = results.get("distances", [[None] * len(ids)])[0]
    chunks = []
    for doc, meta, cid, dist in zip(docs, metas, ids, dists):
        chunk = {
            "chunk_id":   cid,
            "text":       doc,
            "dense_score": 1 - dist if dist is not None else None,
        }
        chunk.update(meta)
        chunks.append(chunk)
    return chunks


# ── Dense retrieval ─────────────────────────────────────────

def dense_retrieve(query: str, corpus: str = "hierarchy", k: int = RETRIEVAL_K) -> list[dict]:
    col   = _collection(corpus)
    q_emb = embed([query])[0]
    res   = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return _format_chroma_results(res)


# ── BM25 sparse retrieval ────────────────────────────────────

def bm25_retrieve(query: str, corpus: str = "hierarchy", k: int = RETRIEVAL_K) -> list[dict]:
    store = get_flat_bm25() if corpus == "flat" else get_hierarchy_bm25()
    return store.search(query, k=k)


# ── Reciprocal Rank Fusion ───────────────────────────────────

def _rrf(rankings: list[list[str]], rrf_k: int = 60) -> list[str]:
    """
    Merge N ordered lists of chunk_ids using RRF.
    Returns a re-ranked list of chunk_ids.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def hybrid_retrieve(query: str, corpus: str = "hierarchy", k: int = RETRIEVAL_K) -> list[dict]:
    """
    Retrieve top-20 from dense + top-20 from BM25, fuse with RRF, return top-k.
    """
    over_k = max(k * 4, 20)   # over-retrieve before fusion
    dense_res = dense_retrieve(query, corpus=corpus, k=over_k)
    bm25_res  = bm25_retrieve( query, corpus=corpus, k=over_k)

    dense_ids = [c["chunk_id"] for c in dense_res]
    bm25_ids  = [c["chunk_id"] for c in bm25_res]
    fused_ids = _rrf([dense_ids, bm25_ids])[:k]

    # look up full chunk dicts — prefer dense metadata, fall back to bm25
    id_to_chunk = {c["chunk_id"]: c for c in bm25_res}
    id_to_chunk.update({c["chunk_id"]: c for c in dense_res})

    return [id_to_chunk[cid] for cid in fused_ids if cid in id_to_chunk]


# ── Public entry point ───────────────────────────────────────

def retrieve(
    query:  str,
    mode:   str = "hybrid",      # "dense" | "bm25" | "hybrid"
    corpus: str = "hierarchy",   # "flat"  | "hierarchy"
    k:      int = RETRIEVAL_K,
) -> list[dict]:
    """
    Returns a list of up to k chunk dicts.
    Each dict has at minimum:
        chunk_id, text, section, breadcrumb, source, chunk_type
    """
    if mode == "dense":
        return dense_retrieve(query, corpus=corpus, k=k)
    elif mode == "bm25":
        return bm25_retrieve(query, corpus=corpus, k=k)
    elif mode == "hybrid":
        return hybrid_retrieve(query, corpus=corpus, k=k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode!r}. "
                         f"Choose from 'dense', 'bm25', 'hybrid'.")


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "What organizations are exempt from tax under cooperative rules?"
    for mode in ("dense", "bm25", "hybrid"):
        results = retrieve(test_query, mode=mode, corpus="hierarchy", k=3)
        print(f"\n── {mode.upper()} ──")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.get('section','')}] {r['text'][:120]}…")
