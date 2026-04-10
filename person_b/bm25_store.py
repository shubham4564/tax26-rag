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
