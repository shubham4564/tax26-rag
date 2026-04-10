"""
The single public interface Person C and Person D import.

Usage:
    from retrieval.retrieve import retrieve

    chunks = retrieve("What cooperative organizations are exempt?",
                      mode="hybrid", corpus="hierarchy", k=5)
"""
from __future__ import annotations
import chromadb
from functools import lru_cache

from config import (
    CHROMA_PATH, FLAT_COLLECTION, HIERARCHY_COLLECTION, RETRIEVAL_K
)
from retrieval.embedder  import embed
from retrieval.bm25_store import get_flat_bm25, get_hierarchy_bm25


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
