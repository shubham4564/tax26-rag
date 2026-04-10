"""
Formats retrieved chunks into a structured context string
and builds the final user prompt.
"""
from person_c.prompts import USER_TEMPLATE, USER_TEMPLATE_NO_RAG


def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into a structured context block.
    Includes index, section number, breadcrumb path, and text.
    """
    if not chunks:
        return "(no context retrieved)"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        section    = chunk.get("section",    "unknown section")
        breadcrumb = chunk.get("breadcrumb", "")
        source     = chunk.get("source",     "federal")
        text       = chunk.get("text",       "").strip()

        lines.append(f"[Source {i} | {source.upper()} | {section}]")
        if breadcrumb and breadcrumb != section:
            lines.append(f"Path: {breadcrumb}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def build_user_prompt(question: str, chunks: list[dict], no_rag: bool = False) -> str:
    if no_rag:
        return USER_TEMPLATE_NO_RAG.format(question=question)
    context = format_context(chunks)
    return USER_TEMPLATE.format(context=context, question=question)
