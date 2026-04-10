"""
Main entry point for Person A.
Run with:  python -m person_a.run_pipeline
"""
import sys
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import DATA_RAW, FLAT_CORPUS_PATH, HIERARCHY_CORPUS_PATH
from shared.utils  import save_jsonl
from person_a.parser        import parse_directory
from person_a.nevada_parser import parse_nevada_directory
from person_a.chunkers      import flat_chunk, hierarchy_chunk


def quality_check(chunks: list, label: str) -> list:
    """Filter bad chunks and print stats."""
    before = len(chunks)
    chunks = [c for c in chunks if 15 <= len(c.text.split()) <= 800]

    # dedup by chunk_id
    seen, deduped = set(), []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    lengths = [len(c.text.split()) for c in deduped]
    avg = sum(lengths) / len(lengths) if lengths else 0
    print(f"[qc] {label}: {before} → {len(deduped)} chunks  avg {avg:.0f} words")
    return deduped


def main():
    # ── 1. Parse federal Title 26 JSON files ──────────────
    federal_raw = DATA_RAW / "federal" / "usc_title26_olrc" / "sections"
    if not federal_raw.exists():
        print(f"[WARN] {federal_raw} not found — skipping federal data")
        federal_nodes = []
    else:
        federal_nodes = parse_directory(federal_raw, source="federal")

    # ── 2. Parse Nevada codes ──────────────────────────────
    nevada_raw = DATA_RAW / "nevada"
    if not nevada_raw.exists():
        print(f"[WARN] {nevada_raw} not found — skipping nevada data")
        nevada_nodes = []
    else:
        nevada_nodes = parse_nevada_directory(nevada_raw)

    all_nodes = federal_nodes + nevada_nodes
    print(f"\n[pipeline] total nodes: {len(all_nodes)}")

    # ── 3. Flat corpus ────────────────────────────────────
    flat_chunks = flat_chunk(all_nodes)
    flat_chunks = quality_check(flat_chunks, "flat")
    save_jsonl([c.to_dict() for c in flat_chunks], FLAT_CORPUS_PATH)
    print(f"[pipeline] saved flat corpus → {FLAT_CORPUS_PATH}")

    # ── 4. Hierarchy corpus ───────────────────────────────
    hier_chunks = hierarchy_chunk(all_nodes)
    hier_chunks = quality_check(hier_chunks, "hierarchy")
    save_jsonl([c.to_dict() for c in hier_chunks], HIERARCHY_CORPUS_PATH)
    print(f"[pipeline] saved hierarchy corpus → {HIERARCHY_CORPUS_PATH}")

    print("\n✓ Person A pipeline complete.")


if __name__ == "__main__":
    main()
