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
