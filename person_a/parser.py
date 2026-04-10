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
