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
