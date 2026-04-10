"""
Core RAG answer function.
Person D imports `answer()` directly.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.prompts        import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_RAG
from generation.llm            import call_llm
from generation.context_builder import build_user_prompt


def answer(
    question:       str,
    retrieval_mode: str = "hybrid",   # "dense" | "bm25" | "hybrid" | "none"
    corpus:         str = "hierarchy",# "flat"  | "hierarchy"
    k:              int = 5,
) -> dict:
    """
    End-to-end RAG pipeline.

    Returns:
        {
          "question":       str,
          "answer":         str,
          "sources":        list[dict],   # retrieved chunks
          "config":         dict,
        }
    """
    no_rag = (retrieval_mode == "none")
    chunks = []

    if not no_rag:
        # Import here to avoid circular imports during testing
        from retrieval.retrieve import retrieve
        chunks = retrieve(question, mode=retrieval_mode, corpus=corpus, k=k)

    sys_prompt  = SYSTEM_PROMPT_NO_RAG if no_rag else SYSTEM_PROMPT
    user_prompt = build_user_prompt(question, chunks, no_rag=no_rag)
    response    = call_llm(sys_prompt, user_prompt)

    return {
        "question": question,
        "answer":   response,
        "sources":  chunks,
        "config": {
            "retrieval_mode": retrieval_mode,
            "corpus":         corpus,
            "k":              k,
            "no_rag":         no_rag,
        },
    }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    q = "Which cooperative organizations are subject to this part of the tax code?"
    for mode, corpus in [
        ("none",   "hierarchy"),
        ("hybrid", "flat"),
        ("hybrid", "hierarchy"),
    ]:
        print(f"\n{'='*60}")
        print(f"Config: mode={mode}, corpus={corpus}")
        result = answer(q, retrieval_mode=mode, corpus=corpus)
        print(result["answer"][:400])
        if result["sources"]:
            print(f"\nSources: {[s['section'] for s in result['sources']]}")
