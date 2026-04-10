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
