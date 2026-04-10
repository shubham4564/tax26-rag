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
