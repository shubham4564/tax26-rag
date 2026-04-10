# ============================================================
# FILE: person_d/load_dataset.py
# ============================================================
"""
Download and prepare the HuggingFace QA eval set.
Run once:  python -m person_d.load_dataset
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from shared.config import EVAL_SET_PATH
from shared.utils  import save_jsonl


# Regex to find section numbers like §162, §1381, §501(c)(3)
SECTION_RE = re.compile(r"§\s*\d[\w().-]*")


def extract_section(text: str) -> str:
    """Return the first section number found in text, else empty string."""
    m = SECTION_RE.search(text or "")
    return m.group(0).replace(" ", "") if m else ""


def prepare_eval_set(max_samples: int = 200) -> list[dict]:
    """
    Load the HuggingFace dataset, normalize fields,
    and extract expected section numbers.
    """
    print("[load_dataset] downloading quotientai/irs_form_instruction_qa_pairs …")
    ds = load_dataset("quotientai/irs_form_instruction_qa_pairs", split="train")
    print(f"[load_dataset] {len(ds)} total records")
    print(f"[load_dataset] columns: {ds.column_names}")

    # --- Normalize field names ---
    # The dataset may use "question"/"answer" or "instruction"/"output" etc.
    # We'll try both naming conventions.
    def get_field(row, *keys):
        for k in keys:
            if k in row and row[k]:
                return str(row[k]).strip()
        return ""

    records = []
    for row in ds:
        question = get_field(row, "question", "instruction", "input", "query")
        answer   = get_field(row, "answer",   "output",      "response", "label")
        if not question or not answer:
            continue

        # Try to find a target section from the question or answer
        expected_section = (
            extract_section(question) or
            extract_section(answer)   or
            ""
        )

        records.append({
            "question":         question,
            "reference_answer": answer,
            "expected_section": expected_section,
            "source_document":  get_field(row, "source", "document", "context", ""),
        })

    # Prefer records that have a known section number (they're easier to evaluate)
    with_section    = [r for r in records if r["expected_section"]]
    without_section = [r for r in records if not r["expected_section"]]

    # Take up to max_samples, biased toward records with sections
    n_with = min(len(with_section), int(max_samples * 0.7))
    n_without = min(len(without_section), max_samples - n_with)
    selected = with_section[:n_with] + without_section[:n_without]

    print(f"[load_dataset] selected {len(selected)} eval records "
          f"({n_with} with section numbers, {n_without} without)")
    return selected


def main():
    records = prepare_eval_set(max_samples=200)
    save_jsonl(records, EVAL_SET_PATH)
    print(f"[load_dataset] saved → {EVAL_SET_PATH}")
    print("\nSample record:")
    import json
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()


# ============================================================
# FILE: person_d/metrics.py
# ============================================================
"""
Retrieval metrics: Recall@K and MRR.
Generation metrics: LLM-as-judge (faithfulness + completeness).
"""
import json, os, re
from dotenv import load_dotenv
load_dotenv()


# ── Retrieval metrics ────────────────────────────────────────

def recall_at_k(expected_section: str, retrieved_chunks: list[dict], k: int = 5) -> float:
    """
    1.0 if expected_section appears in the section field of any top-k chunk.
    0.0 if expected_section is empty (unverifiable) — returns NaN sentinel.
    """
    if not expected_section:
        return float("nan")
    top_k_sections = [c.get("section", "") for c in retrieved_chunks[:k]]
    # Normalize comparison: "§1381" matches "§ 1381" etc.
    norm = lambda s: re.sub(r"\s", "", s).upper()
    exp  = norm(expected_section)
    return float(any(norm(s) == exp for s in top_k_sections))


def mrr(expected_section: str, retrieved_chunks: list[dict], k: int = 10) -> float:
    """Mean Reciprocal Rank. Returns NaN if expected_section is empty."""
    if not expected_section:
        return float("nan")
    norm = lambda s: re.sub(r"\s", "", s).upper()
    exp  = norm(expected_section)
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if norm(chunk.get("section", "")) == exp:
            return 1.0 / rank
    return 0.0


# ── LLM-as-judge ─────────────────────────────────────────────

JUDGE_SYSTEM = "You are a precise evaluator of tax question-answering systems. \
Always respond with valid JSON only."

JUDGE_TEMPLATE = """Evaluate the following tax QA response.

Question: {question}

Reference answer: {reference}

System answer: {prediction}

Score on two dimensions (each 1–5):
1. faithfulness  — Is every claim in the system answer grounded in tax law \
(no hallucinated section numbers or invented rules)?
   1=mostly hallucinated, 3=some errors, 5=fully grounded
2. completeness  — Does the system answer cover all key points in the reference?
   1=most points missing, 3=partial, 5=covers all key points

Respond ONLY with a JSON object. No other text. Example:
{{"faithfulness": 4, "completeness": 3}}
"""


def llm_judge(
    question:   str,
    reference:  str,
    prediction: str,
    provider:   str | None = None,
) -> dict:
    """
    Returns {"faithfulness": 1-5, "completeness": 1-5}.
    Falls back to {"faithfulness": -1, "completeness": -1} on error.
    """
    from person_c.llm import call_llm
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        reference=reference[:800],      # truncate to save tokens
        prediction=prediction[:800],
    )
    try:
        raw = call_llm(JUDGE_SYSTEM, prompt, temperature=0.0, max_tokens=64)
        # Strip possible markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)
        return {
            "faithfulness":  int(scores.get("faithfulness",  -1)),
            "completeness":  int(scores.get("completeness",  -1)),
        }
    except Exception as e:
        print(f"  [judge WARN] {e} | raw={raw!r}")
        return {"faithfulness": -1, "completeness": -1}


# ============================================================
# FILE: person_d/run_eval.py
# ============================================================
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


# ============================================================
# FILE: person_d/analyze.py
# ============================================================
"""
Load eval results and produce the comparison table + findings.
Usage:  python -m person_d.analyze
        python -m person_d.analyze --path outputs/eval_results/eval_20240101_120000.csv
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pandas as pd
from shared.config import OUTPUTS


def nanmean(series: pd.Series) -> float:
    vals = [v for v in series if not (isinstance(v, float) and math.isnan(v)) and v != -1]
    return sum(vals) / len(vals) if vals else float("nan")


def print_comparison_table(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print(f"{'Config':<25} {'Recall@5':>10} {'MRR':>8} {'Faithful':>10} {'Complete':>10}")
    print("=" * 72)

    config_names = [
        ("no_rag",           "No RAG (baseline)"),
        ("flat_hybrid",      "Flat + Hybrid"),
        ("hierarchy_dense",  "Hierarchy + Dense"),
        ("hierarchy_hybrid", "Hierarchy + Hybrid"),
    ]

    summary_rows = []
    for name, label in config_names:
        r5   = nanmean(df.get(f"{name}_recall_at_5", pd.Series([float("nan")])))
        mrr_ = nanmean(df.get(f"{name}_mrr",         pd.Series([float("nan")])))
        faith = nanmean(df.get(f"{name}_faithfulness", pd.Series([])))
        comp  = nanmean(df.get(f"{name}_completeness", pd.Series([])))

        r5_s   = f"{r5:.3f}"    if not math.isnan(r5)    else "   n/a"
        mrr_s  = f"{mrr_:.3f}"  if not math.isnan(mrr_)  else "   n/a"
        faith_s = f"{faith:.2f}/5" if not math.isnan(faith) else "   n/a"
        comp_s  = f"{comp:.2f}/5"  if not math.isnan(comp)  else "   n/a"

        print(f"{label:<25} {r5_s:>10} {mrr_s:>8} {faith_s:>10} {comp_s:>10}")
        summary_rows.append({
            "config": label,
            "recall_at_5": round(r5, 3)   if not math.isnan(r5)    else None,
            "mrr":         round(mrr_, 3) if not math.isnan(mrr_)  else None,
            "faithfulness":round(faith,2) if not math.isnan(faith) else None,
            "completeness":round(comp, 2) if not math.isnan(comp)  else None,
        })

    print("=" * 72)
    return pd.DataFrame(summary_rows)


def find_interesting_cases(df: pd.DataFrame, n: int = 5):
    """
    Show cases where hierarchy_hybrid beat flat_hybrid most and least.
    """
    hier_faith = df.get("hierarchy_hybrid_faithfulness", pd.Series())
    flat_faith = df.get("flat_hybrid_faithfulness",      pd.Series())
    if hier_faith.empty or flat_faith.empty:
        return

    df2 = df.copy()
    df2["faith_delta"] = hier_faith.fillna(0) - flat_faith.fillna(0)

    print(f"\n── Top {n} cases where hierarchy_hybrid BEAT flat_hybrid ──")
    top = df2.nlargest(n, "faith_delta")[["question", "faith_delta",
                                          "hierarchy_hybrid_answer",
                                          "flat_hybrid_answer"]]
    for _, row in top.iterrows():
        print(f"\n  Q: {row['question'][:100]}")
        print(f"  Delta: +{row['faith_delta']:.1f}")
        print(f"  Hierarchy answer: {str(row['hierarchy_hybrid_answer'])[:150]}…")

    print(f"\n── Top {n} cases where hierarchy_hybrid DID NOT help ──")
    bottom = df2.nsmallest(n, "faith_delta")[["question", "faith_delta",
                                               "hierarchy_hybrid_answer",
                                               "flat_hybrid_answer"]]
    for _, row in bottom.iterrows():
        print(f"\n  Q: {row['question'][:100]}")
        print(f"  Delta: {row['faith_delta']:.1f}")
        print(f"  Hierarchy answer: {str(row['hierarchy_hybrid_answer'])[:150]}…")


def write_markdown_report(summary_df: pd.DataFrame, out_path: Path):
    lines = [
        "# Tax RAG Evaluation Report",
        "",
        "## Comparison Table",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Key Findings",
        "",
        "*(Fill in after reviewing the comparison table and interesting cases.)*",
        "",
        "### Where hierarchy-aware chunking helped",
        "- TODO",
        "",
        "### Where hybrid retrieval added value over dense-only",
        "- TODO",
        "",
        "### No-RAG baseline failure modes",
        "- TODO",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[analyze] report saved → {out_path}")


def main(path: str | None = None):
    csv_path = Path(path) if path else OUTPUTS / "eval_latest.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run person_d.run_eval first.")
        return

    df = pd.read_csv(csv_path)
    print(f"[analyze] loaded {len(df)} rows from {csv_path}")

    summary_df = print_comparison_table(df)
    find_interesting_cases(df, n=5)

    report_path = OUTPUTS / "final_report.md"
    write_markdown_report(summary_df, report_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None,
                   help="Path to a specific CSV (default: eval_latest.csv)")
    args = p.parse_args()
    main(path=args.path)
