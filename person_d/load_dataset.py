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
