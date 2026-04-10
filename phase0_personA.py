# ============================================================
# FILE: requirements.txt
# ============================================================
# chromadb
# rank_bm25
# sentence-transformers
# openai
# anthropic
# streamlit
# datasets
# pandas
# tqdm
# python-dotenv
# tiktoken

# ============================================================
# FILE: shared/schema.py
# ============================================================
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class ChunkMetadata:
    title:      str = ""
    subtitle:   str = ""
    chapter:    str = ""
    subchapter: str = ""
    part:       str = ""
    heading:    str = ""

@dataclass
class Chunk:
    chunk_id:   str = ""
    text:       str = ""
    source:     str = ""      # "federal" | "nevada"
    section:    str = ""      # e.g. "§1381"
    breadcrumb: str = ""      # e.g. "Title 26 > §1381 > (a) > (2) > (B)"
    chunk_type: str = ""      # "flat" | "hierarchy"
    metadata:   ChunkMetadata = field(default_factory=ChunkMetadata)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = asdict(self.metadata)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        meta = ChunkMetadata(**d.pop("metadata", {}))
        return Chunk(**d, metadata=meta)


# ============================================================
# FILE: shared/config.py
# ============================================================
import os
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_CHUNKS = ROOT / "data" / "chunks"
DATA_INDEX  = ROOT / "data" / "indexes"
DATA_EVAL   = ROOT / "data"
OUTPUTS     = ROOT / "outputs" / "eval_results"

FLAT_CORPUS_PATH      = DATA_CHUNKS / "flat_corpus.jsonl"
HIERARCHY_CORPUS_PATH = DATA_CHUNKS / "hierarchy_corpus.jsonl"
EVAL_SET_PATH         = DATA_EVAL   / "eval_set.jsonl"

EMBED_MODEL  = "BAAI/bge-large-en-v1.5"   # local, free
LLM_MODEL    = "gpt-4o-mini"               # swap to "claude-3-haiku-20240307" if using Anthropic
CHROMA_PATH  = str(DATA_INDEX)

FLAT_COLLECTION      = "flat_corpus"
HIERARCHY_COLLECTION = "hierarchy_corpus"

RETRIEVAL_K  = 5
MAX_TOKENS   = 400   # target chunk size (words, ~tokens)

for p in [DATA_RAW, DATA_CHUNKS, DATA_INDEX, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE: shared/utils.py
# ============================================================
import json
from pathlib import Path
from typing import Iterator

def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def batched(lst: list, n: int) -> Iterator[list]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ============================================================
# FILE: person_a/parser.py
# ============================================================
"""
Recursively walks a statute JSON node tree and yields flat
leaf-level dicts ready for chunking.
"""
import json
import re
from pathlib import Path
from typing import Iterator

# --------------- helpers ---------------

def _table_to_text(table: list[dict] | None) -> str:
    """Serialize a TaxTableRow list to a readable string."""
    if not table:
        return ""
    lines = []
    for row in table:
        rng = row.get("income_range", "")
        frm = row.get("tax_formula", "")
        lines.append(f"{rng}: {frm}")
    return "\n".join(lines)

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _make_id(section: str, breadcrumb: str) -> str:
    """Create a deterministic, filesystem-safe chunk ID."""
    crumb = re.sub(r"[^a-zA-Z0-9_]", "_", breadcrumb)
    return f"{section}_{crumb}"[:120]

# --------------- recursive extractor ---------------

def extract_nodes(
    node: dict,
    breadcrumb_parts: list[str],
    section_number: str,
    source: str,
    usc_meta: dict,
) -> Iterator[dict]:
    """
    Yields one dict per node that carries text content.
    breadcrumb_parts accumulates labels from root to current node.
    """
    label   = node.get("label", "")
    heading = _clean(node.get("heading", ""))
    text    = _clean(node.get("text",    ""))
    table   = node.get("table")

    current_parts = breadcrumb_parts + ([label] if label else [])
    breadcrumb    = " > ".join(p for p in current_parts if p)

    # Combine heading + text + table into one content string
    parts = []
    if heading:
        parts.append(heading)
    if text:
        parts.append(text)
    if table:
        parts.append(_table_to_text(table))
    content = " ".join(parts)

    if content:
        yield {
            "raw_text":   content,
            "breadcrumb": breadcrumb,
            "label":      label,
            "heading":    heading,
            "section":    section_number,
            "source":     source,
            "metadata":   usc_meta,
        }

    for child in node.get("children", []):
        yield from extract_nodes(
            child, current_parts, section_number, source, usc_meta
        )


def parse_section(data: dict, source: str = "federal") -> list[dict]:
    """
    Top-level entry point. Takes one scraped section JSON dict,
    returns a list of extracted node dicts.
    """
    if data.get("section_type", "normal") != "normal":
        return []

    usc = data.get("metadata", {}).get("usc_location", {})
    section_number = usc.get("section", "§?")

    usc_meta = {
        "title":      usc.get("title",      ""),
        "subtitle":   usc.get("subtitle",   ""),
        "chapter":    usc.get("chapter",    ""),
        "subchapter": usc.get("subchapter", ""),
        "part":       usc.get("part",       ""),
        "heading":    data.get("statute", {}).get("heading", ""),
    }

    statute = data.get("statute", {})
    # seed breadcrumb with section number
    nodes = list(extract_nodes(
        statute,
        breadcrumb_parts=[section_number],
        section_number=section_number,
        source=source,
        usc_meta=usc_meta,
    ))
    return nodes


def parse_directory(raw_dir: Path, source: str = "federal") -> list[dict]:
    """Parse every .json file in raw_dir and return all extracted nodes."""
    all_nodes = []
    files = sorted(raw_dir.glob("*.json"))
    print(f"[parser] found {len(files)} files in {raw_dir}")
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            all_nodes.extend(parse_section(data, source=source))
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}")
    print(f"[parser] extracted {len(all_nodes)} nodes total")
    return all_nodes


# ============================================================
# FILE: person_a/chunkers.py
# ============================================================
"""
Two chunking strategies:
  1. flat_chunk   — merge + split by token count, no hierarchy signal
  2. hierarchy_chunk — prepend full breadcrumb path to every chunk text
                       + add a parent-summary chunk per subsection
"""
import hashlib
from shared.schema import Chunk, ChunkMetadata

MAX_WORDS = 400   # ~400 tokens


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split a text string into ≤max_words segments at sentence boundaries."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, buf = [], []
    buf_len = 0
    for sent in sentences:
        w = _word_count(sent)
        if buf_len + w > max_words and buf:
            chunks.append(". ".join(buf) + ".")
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += w
    if buf:
        chunks.append(". ".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _make_chunk_id(text: str, prefix: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    safe = prefix.replace(" ", "_").replace(">", "").replace("§", "S")[:60]
    return f"{safe}_{h}"


def _meta_from_node(node: dict) -> ChunkMetadata:
    m = node.get("metadata", {})
    return ChunkMetadata(
        title=m.get("title", ""),
        subtitle=m.get("subtitle", ""),
        chapter=m.get("chapter", ""),
        subchapter=m.get("subchapter", ""),
        part=m.get("part", ""),
        heading=m.get("heading", ""),
    )


# ── FLAT CHUNKER ────────────────────────────────────────────

def flat_chunk(nodes: list[dict]) -> list[Chunk]:
    """
    Merge consecutive small nodes from the same section,
    split oversized ones. No breadcrumb prepended.
    """
    chunks = []
    buf_text, buf_meta, buf_section, buf_source = "", None, "", ""

    def flush():
        nonlocal buf_text
        if not buf_text.strip():
            return
        for seg in _split_long(buf_text):
            if _word_count(seg) < 15:
                continue
            cid = _make_chunk_id(seg, buf_section)
            chunks.append(Chunk(
                chunk_id=cid,
                text=seg,
                source=buf_source,
                section=buf_section,
                breadcrumb=buf_section,
                chunk_type="flat",
                metadata=buf_meta or ChunkMetadata(),
            ))
        buf_text = ""

    for node in nodes:
        if node["section"] != buf_section:
            flush()
            buf_section = node["section"]
            buf_source  = node["source"]
            buf_meta    = _meta_from_node(node)

        words = _word_count(node["raw_text"])
        if _word_count(buf_text) + words > MAX_WORDS:
            flush()

        buf_text += " " + node["raw_text"]

    flush()
    return chunks


# ── HIERARCHY CHUNKER ────────────────────────────────────────

def hierarchy_chunk(nodes: list[dict]) -> list[Chunk]:
    """
    1. Each leaf node becomes its own chunk with breadcrumb prepended.
    2. Additionally creates a 'parent summary' chunk per unique
       top-level subsection (groups all children under (a), (b), etc.)
    """
    chunks = []
    parent_groups: dict[str, list[dict]] = {}  # key: section + top_label

    for node in nodes:
        breadcrumb = node["breadcrumb"]
        # Prepend the breadcrumb so embeddings capture legal context
        prefixed_text = f"{breadcrumb}: {node['raw_text']}"
        segments = _split_long(prefixed_text)

        for seg in segments:
            if _word_count(seg) < 15:
                continue
            cid = _make_chunk_id(seg, breadcrumb)
            chunks.append(Chunk(
                chunk_id=cid,
                text=seg,
                source=node["source"],
                section=node["section"],
                breadcrumb=breadcrumb,
                chunk_type="hierarchy",
                metadata=_meta_from_node(node),
            ))

        # group by "section + first-level label" for parent summaries
        parts = breadcrumb.split(" > ")
        if len(parts) >= 2:
            key = f"{parts[0]}_{parts[1]}"
            parent_groups.setdefault(key, []).append(node)

    # ── parent summary chunks ──
    for key, group_nodes in parent_groups.items():
        combined = " ".join(n["raw_text"] for n in group_nodes)
        if _word_count(combined) < 20:
            continue
        first = group_nodes[0]
        # take first 500 words as the summary chunk
        summary_text = " ".join(combined.split()[:500])
        crumb = first["breadcrumb"].rsplit(" > ", 1)[0]  # parent path
        prefixed = f"{crumb} [summary]: {summary_text}"
        cid = _make_chunk_id(prefixed, crumb + "_summary")
        chunks.append(Chunk(
            chunk_id=cid,
            text=prefixed,
            source=first["source"],
            section=first["section"],
            breadcrumb=crumb + " [summary]",
            chunk_type="hierarchy",
            metadata=_meta_from_node(first),
        ))

    return chunks


# ============================================================
# FILE: person_a/nevada_parser.py
# ============================================================
"""
Minimal parser for Nevada tax codes.
Adjust `parse_nevada_file` to match your actual Nevada file format.
"""
import json, re
from pathlib import Path


def parse_nevada_file(fp: Path) -> list[dict]:
    """
    Adapt this to your Nevada data format.
    Returns a list of node dicts compatible with the Title 26 schema.
    Expected Nevada JSON shape (minimal):
      { "section": "NRS 360.xxx", "heading": "...", "text": "..." }
    """
    data = json.loads(fp.read_text(encoding="utf-8"))

    # If Nevada data is a flat list of sections:
    if isinstance(data, list):
        nodes = []
        for item in data:
            section = item.get("section", "NRS ?")
            heading = item.get("heading", "")
            text    = re.sub(r"\s+", " ", item.get("text", "")).strip()
            if text:
                nodes.append({
                    "raw_text":   f"{heading} {text}".strip(),
                    "breadcrumb": section,
                    "label":      section,
                    "heading":    heading,
                    "section":    section,
                    "source":     "nevada",
                    "metadata": {
                        "title": "Nevada Revised Statutes",
                        "subtitle": "", "chapter": "", "subchapter": "",
                        "part": "", "heading": heading,
                    },
                })
        return nodes

    # If Nevada data is a single section dict (same shape as Title 26):
    if isinstance(data, dict) and "statute" in data:
        from person_a.parser import parse_section
        return parse_section(data, source="nevada")

    return []


def parse_nevada_directory(nevada_dir: Path) -> list[dict]:
    all_nodes = []
    for fp in sorted(nevada_dir.glob("*.json")):
        try:
            all_nodes.extend(parse_nevada_file(fp))
        except Exception as e:
            print(f"  [WARN nevada] {fp.name}: {e}")
    print(f"[nevada_parser] extracted {len(all_nodes)} nodes")
    return all_nodes


# ============================================================
# FILE: person_a/run_pipeline.py
# ============================================================
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
    federal_raw = DATA_RAW / "federal"
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
