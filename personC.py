# ============================================================
# FILE: person_c/prompts.py
# ============================================================

SYSTEM_PROMPT = """You are a precise tax code assistant for the United States \
Internal Revenue Code (Title 26) and Nevada state tax law.

Rules you must follow in every response:
1. Ground every statement in the provided tax code context. \
   If a claim is not supported by the context, say so explicitly.
2. Cite the exact section number (e.g., "Under §1381(a)(2)…") \
   for every legal point you make.
3. When both federal (Title 26) and Nevada provisions appear, \
   clearly label which jurisdiction each point applies to.
4. If the context does not contain enough information to answer, \
   say: "I cannot find a relevant provision in the provided context \
   for this question." Do not guess or invent rules.
5. Never give definitive legal advice. Frame answers as \
   "the code states…" or "under §X, …" — not "you should…".
6. Be concise. Use plain English where possible.
"""

# Used when no context is available (no-RAG baseline)
SYSTEM_PROMPT_NO_RAG = """You are a tax code assistant for US federal \
and Nevada state tax law. Answer based on your general knowledge. \
Clearly label any uncertainty. Do not invent section numbers."""

USER_TEMPLATE = """\
Using ONLY the tax code provisions below, answer the question.
Cite the specific section number for each point you make.
If the context does not cover the question, say so clearly.

--- TAX CODE CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
"""

USER_TEMPLATE_NO_RAG = """\
Answer the following tax question based on your general knowledge.
Note any uncertainty.

Question: {question}
"""


# ============================================================
# FILE: person_c/llm.py
# ============================================================
"""
Thin wrapper around LLM APIs.
Set LLM_PROVIDER in .env to "openai" (default) or "anthropic".
"""
import os
from dotenv import load_dotenv
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


def call_llm(
    system_prompt: str,
    user_message:  str,
    model:         str | None = None,
    temperature:   float = 0.0,
    max_tokens:    int = 1024,
) -> str:
    """
    Call the configured LLM and return the text response.
    temperature=0.0 for deterministic, grounded answers.
    """
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message,
                               model or "claude-3-haiku-20240307",
                               temperature, max_tokens)
    else:
        return _call_openai(system_prompt, user_message,
                            model or "gpt-4o-mini",
                            temperature, max_tokens)


def _call_openai(system_prompt, user_message, model, temperature, max_tokens) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(system_prompt, user_message, model, temperature, max_tokens) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text.strip()


# ============================================================
# FILE: person_c/context_builder.py
# ============================================================
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


# ============================================================
# FILE: person_c/pipeline.py
# ============================================================
"""
Core RAG answer function.
Person D imports `answer()` directly.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from person_c.prompts        import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_RAG
from person_c.llm            import call_llm
from person_c.context_builder import build_user_prompt


def answer(
    question:       str,
    retrieval_mode: str = "hybrid",   # "dense" | "bm25" | "hybrid" | "none"
    corpus:         str = "hierarchy",# "flat"  | "hierarchy"
    k:              int = 5,
) -> dict:
    """
    End-to-end RAG pipeline.

    Returns:
        {
          "question":       str,
          "answer":         str,
          "sources":        list[dict],   # retrieved chunks
          "config":         dict,
        }
    """
    no_rag = (retrieval_mode == "none")
    chunks = []

    if not no_rag:
        # Import here to avoid circular imports during testing
        from person_b.retrieve import retrieve
        chunks = retrieve(question, mode=retrieval_mode, corpus=corpus, k=k)

    sys_prompt  = SYSTEM_PROMPT_NO_RAG if no_rag else SYSTEM_PROMPT
    user_prompt = build_user_prompt(question, chunks, no_rag=no_rag)
    response    = call_llm(sys_prompt, user_prompt)

    return {
        "question": question,
        "answer":   response,
        "sources":  chunks,
        "config": {
            "retrieval_mode": retrieval_mode,
            "corpus":         corpus,
            "k":              k,
            "no_rag":         no_rag,
        },
    }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    q = "Which cooperative organizations are subject to this part of the tax code?"
    for mode, corpus in [
        ("none",   "hierarchy"),
        ("hybrid", "flat"),
        ("hybrid", "hierarchy"),
    ]:
        print(f"\n{'='*60}")
        print(f"Config: mode={mode}, corpus={corpus}")
        result = answer(q, retrieval_mode=mode, corpus=corpus)
        print(result["answer"][:400])
        if result["sources"]:
            print(f"\nSources: {[s['section'] for s in result['sources']]}")


# ============================================================
# FILE: person_c/app.py
# ============================================================
"""
Streamlit demo app.
Run with:  streamlit run person_c/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from person_c.pipeline import answer

st.set_page_config(page_title="Tax Code RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Tax Code Assistant")
st.caption("Powered by Title 26 (IRC) + Nevada Tax Codes — RAG system")

# ── Sidebar: config ──────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    no_rag = st.checkbox("No-RAG baseline (LLM only, no retrieval)", value=False)

    corpus = st.radio(
        "Corpus (chunking strategy)",
        options=["hierarchy", "flat"],
        index=0,
        disabled=no_rag,
    )
    retrieval_mode = st.radio(
        "Retrieval mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        disabled=no_rag,
    )
    k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5,
                  disabled=no_rag)

    st.divider()
    st.markdown("**Active config**")
    if no_rag:
        st.info("No RAG — pure LLM response")
    else:
        st.success(f"Corpus: `{corpus}`\nRetrieval: `{retrieval_mode}`\nK: `{k}`")

# ── Main area: question input ────────────────────────────────
question = st.text_area(
    "Ask a tax question",
    placeholder="e.g. Which cooperative organizations are exempt under §1381?",
    height=100,
)

col1, col2 = st.columns([1, 5])
submit = col1.button("Submit", type="primary")
clear  = col2.button("Clear")

if clear:
    st.rerun()

if submit and question.strip():
    with st.spinner("Retrieving and generating …"):
        mode   = "none" if no_rag else retrieval_mode
        result = answer(question, retrieval_mode=mode, corpus=corpus, k=k)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["sources"]:
        with st.expander(f"📄 Sources ({len(result['sources'])} chunks retrieved)"):
            for i, src in enumerate(result["sources"], 1):
                section    = src.get("section",    "")
                breadcrumb = src.get("breadcrumb", "")
                source_tag = src.get("source",     "federal").upper()
                text       = src.get("text",       "")

                st.markdown(f"**[{i}] {source_tag} — {section}**")
                st.caption(breadcrumb)
                st.text(text[:400] + ("…" if len(text) > 400 else ""))
                st.divider()

    st.divider()
    with st.expander("Raw result dict"):
        st.json({
            "question": result["question"],
            "config":   result["config"],
            "answer":   result["answer"],
            "num_sources": len(result["sources"]),
        })

elif submit and not question.strip():
    st.warning("Please enter a question.")
