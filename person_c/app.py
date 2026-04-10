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
