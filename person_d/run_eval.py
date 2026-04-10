"""
Run the full evaluation across all 4 system configurations.
Usage:
    python -m person_d.run_eval           # full run
    python -m person_d.run_eval --dry-run # 5 questions only
    python -m person_d.run_eval --n 50    # custom sample size
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from shared.config import EVAL_SET_PATH, OUTPUTS
from shared.utils  import load_jsonl
from person_c.pipeline import answer
from person_b.retrieve  import retrieve
from person_d.metrics   import recall_at_k, mrr, llm_judge


# ── System configurations to compare ────────────────────────
CONFIGS = [
    {
        "name":           "no_rag",
        "retrieval_mode": "none",
        "corpus":         "hierarchy",
        "description":    "No retrieval — pure LLM",
    },
    {
        "name":           "flat_hybrid",
        "retrieval_mode": "hybrid",
        "corpus":         "flat",
        "description":    "Flat chunking + hybrid retrieval",
    },
    {
        "name":           "hierarchy_dense",
        "retrieval_mode": "dense",
        "corpus":         "hierarchy",
        "description":    "Hierarchy chunking + dense retrieval",
    },
    {
        "name":           "hierarchy_hybrid",
        "retrieval_mode": "hybrid",
        "corpus":         "hierarchy",
        "description":    "Hierarchy chunking + hybrid retrieval (best expected)",
    },
]


def eval_one(record: dict) -> dict:
    """
    Run all 4 configs on a single QA record.
    Returns a flat dict of all scores for this question.
    """
    q         = record["question"]
    ref       = record["reference_answer"]
    exp_sec   = record.get("expected_section", "")

    row = {
        "question":         q,
        "reference_answer": ref,
        "expected_section": exp_sec,
    }

    for cfg in CONFIGS:
        name = cfg["name"]
        try:
            result = answer(q,
                            retrieval_mode=cfg["retrieval_mode"],
                            corpus=cfg["corpus"],
                            k=5)
            pred   = result["answer"]
            chunks = result["sources"]
        except Exception as e:
            print(f"  [ERROR] config={name} q={q[:60]!r}: {e}")
            pred, chunks = "", []

        row[f"{name}_answer"] = pred

        # Retrieval metrics (only meaningful for non-no_rag configs)
        if cfg["retrieval_mode"] != "none":
            row[f"{name}_recall_at_5"] = recall_at_k(exp_sec, chunks, k=5)
            row[f"{name}_mrr"]         = mrr(exp_sec, chunks, k=10)
        else:
            row[f"{name}_recall_at_5"] = float("nan")
            row[f"{name}_mrr"]         = float("nan")

        # Generation metrics (LLM judge)
        scores = llm_judge(q, ref, pred)
        row[f"{name}_faithfulness"] = scores["faithfulness"]
        row[f"{name}_completeness"] = scores["completeness"]

    return row


def main(dry_run: bool = False, n: int | None = None):
    records = load_jsonl(EVAL_SET_PATH)

    if dry_run:
        records = records[:5]
        print("[run_eval] DRY RUN — 5 questions only")
    elif n:
        records = records[:n]
        print(f"[run_eval] running on {n} questions")
    else:
        print(f"[run_eval] running on {len(records)} questions")

    print(f"[run_eval] configs: {[c['name'] for c in CONFIGS]}")

    rows = []
    for record in tqdm(records, desc="evaluating"):
        rows.append(eval_one(record))

    df = pd.DataFrame(rows)

    # Save full results
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUTS / f"eval_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[run_eval] saved → {out_path}")

    # Also save as latest (for analyze.py)
    latest_path = OUTPUTS / "eval_latest.csv"
    df.to_csv(latest_path, index=False)
    print(f"[run_eval] saved → {latest_path}")

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n",       type=int, default=None)
    args = p.parse_args()
    main(dry_run=args.dry_run, n=args.n)
