# Tax RAG System — Implementation Plan

---

## Phase 0: Shared Setup (Day 1 — All Four, ~2 hours together)

Do this as a group before anyone splits off. Getting this wrong causes integration problems later.

**Step 0.1 — Create a shared Git repo**
One person creates it, everyone clones. Agree on a branch strategy: `main` is stable, each person works on `person-a/feature-name`. No one pushes directly to main until integration week.

**Step 0.2 — Agree on the shared data contract**
This is the most important decision you make on day 1. Every person's code depends on the shape of a "chunk." Agree on this exact structure before anyone writes a line:

```python
{
  "chunk_id":   "§1381_a_2_B",          # unique, deterministic ID
  "text":       "...",                    # the actual content Person B embeds
  "source":     "federal" | "nevada",
  "section":    "§1381",
  "breadcrumb": "Title 26 > §1381 > (a) > (2) > (B)",
  "chunk_type": "flat" | "hierarchy",    # which corpus this belongs to
  "metadata": {
    "title":      "Title 26 – Internal Revenue Code",
    "subtitle":   "Subtitle A",
    "chapter":    "CHAPTER 1",
    "subchapter": "Subchapter T",
    "heading":    "Organizations to which part applies"
  }
}
```

Write this as a Python dataclass or TypedDict in `shared/schema.py`. Everyone imports from there.

**Step 0.3 — Set up the Python environment**
Create a shared `requirements.txt`:

```
langchain-community
chromadb
rank_bm25
sentence-transformers
openai          # or anthropic
streamlit
datasets        # HuggingFace
evaluate
pandas
tqdm
python-dotenv
```

Everyone runs `pip install -r requirements.txt` in a fresh virtualenv.

**Step 0.4 — Set up folder structure**

```
tax-rag/
  shared/
    schema.py        # chunk dataclass (agreed in Step 0.2)
    config.py        # shared constants: paths, model names, k values
  person_a/          # data engineering
  person_b/          # indexing & retrieval
  person_c/          # generation & pipeline
  person_d/          # evaluation
  data/
    raw/             # A's input: scraped JSON files + Nevada codes
    chunks/          # A's output: flat_corpus.jsonl, hierarchy_corpus.jsonl
    indexes/         # B's output: chroma DBs
  outputs/
    eval_results/    # D's output tables
```

**Step 0.5 — Smoke test**
Each person writes a 5-line "hello world" that imports from `shared/schema.py` and prints a dummy chunk. If everyone's import works, setup is done.

---

## Phase 1A: Data Engineering (Person A, Days 1–7)

### Step 1 — Understand the input data (Day 1, ~3 hours)

Before writing any code, manually inspect 5–10 of your scraped JSON files. Specifically check:

- How deep does the `statute.children` nesting go in practice? (The schema supports up to subclause level.)
- Are there sections with `"section_type": "repealed"` or `"reserved"`? You'll skip these.
- Are there sections where `statute.text` has useful content at the root level (not just in children)?
- What does the Nevada tax code data look like? Same JSON schema or different format?

Write your observations in a `person_a/NOTES.md`. This saves debugging time later.

### Step 2 — Write the recursive parser (Days 1–2)

Create `person_a/parser.py`. Write a recursive function that walks the `statute` tree depth-first:

```python
def extract_nodes(node, breadcrumb_parts, section_number, source):
    """
    Recursively walk a statute node and yield leaf-level text chunks.
    breadcrumb_parts: list of label strings accumulated on the way down
    """
    current_path = breadcrumb_parts + [node.get("label", "")]
    text = node.get("text", "").strip()

    if text:
        yield {
            "text": text,
            "breadcrumb": " > ".join(p for p in current_path if p),
            "section": section_number,
            "source": source,
            # ... rest of schema fields
        }

    for child in node.get("children", []):
        yield from extract_nodes(child, current_path, section_number, source)
```

Important edge cases to handle:
- Nodes where `text` is empty but children exist (common at subsection level) — recurse but don't yield
- Nodes with a `table` field (tax rate tables like in §1(a)–(e)) — serialize the table rows to a string and include them
- `section_type != "normal"` — skip entirely

Test this on 10 sections manually. Print the output and verify the breadcrumbs look correct before moving on.

### Step 3 — Build the flat chunker (Day 2)

Create `person_a/chunkers.py`. The flat chunker takes the extracted nodes from Step 2 and merges + splits them into ~400-token chunks with no hierarchy context:

```python
def flat_chunk(nodes, max_tokens=400):
    """
    Concatenate node texts and split at token boundaries.
    Strip breadcrumb — just raw text + minimal metadata.
    """
```

Use a simple word-count proxy for tokens (1 token ≈ 0.75 words) unless you want to install `tiktoken` for precision. For the baseline, exact token counts don't matter much.

Each output chunk gets `"chunk_type": "flat"`.

### Step 4 — Build the hierarchy-aware chunker (Days 3–4)

This is the more important chunker. The key idea: each chunk's `text` field should contain the breadcrumb path as a prepended header, so the embedding captures legal context, not just the leaf text.

```python
def hierarchy_chunk(node, parent_context=""):
    """
    For each leaf node, prepend the full breadcrumb + parent heading text.
    This means the embedding sees:
      '§1381 > (a) > (2) > (B): which is subject to the provisions of...'
    rather than just:
      'which is subject to the provisions of...'
    """
    context_prefix = f"{node['breadcrumb']}: "
    full_text = context_prefix + node['text']
    return full_text
```

Additionally, for every leaf node also create a "parent summary chunk" — a chunk that contains the subsection heading + the concatenated text of all its direct children (capped at 500 tokens). This gives the retriever a coarser-grained option when the query is broad.

Each output chunk gets `"chunk_type": "hierarchy"`.

### Step 5 — Handle Nevada tax codes (Day 4)

The Nevada codes are likely in a different format. Write a separate `person_a/nevada_parser.py`. Even if the format is messier, get at minimum:

- Section number
- Section heading
- Body text

Tag every Nevada chunk with `"source": "nevada"`. If the format is very different from Title 26, just do simple paragraph splitting — don't force the Title 26 hierarchy logic onto it.

### Step 6 — Run over the full corpus and export (Days 5–6)

Write `person_a/run_pipeline.py`. This should:

1. Iterate over every JSON file in `data/raw/`
2. Skip `section_type != "normal"`
3. Run both chunkers
4. Write two output files:
   - `data/chunks/flat_corpus.jsonl` (one JSON object per line)
   - `data/chunks/hierarchy_corpus.jsonl`

Use `tqdm` for a progress bar. Log how many chunks each file produces. After the full run, print summary stats:

```
flat_corpus:       12,483 chunks  avg length: 312 tokens
hierarchy_corpus:  18,941 chunks  avg length: 287 tokens
```

The hierarchy corpus should be larger because of the parent summary chunks.

### Step 7 — Quality check (Day 7)

Before handing off to Person B, do a sanity pass:

- Sample 20 random chunks from each corpus and read them. Do the breadcrumbs look right?
- Check for any chunks under 20 tokens (likely a parsing artifact — filter them out)
- Check for any chunks over 800 tokens (split them)
- Verify that every chunk has all fields from the agreed schema (Step 0.2)
- Run `assert all(c["chunk_id"] for c in corpus)` — every chunk must have a unique ID

Hand off: copy both `.jsonl` files to `data/chunks/` in the shared repo and announce in the group chat.

---

## Phase 1B: Indexing & Retrieval (Person B, Days 1–7)

### Step 1 — Set up Chroma locally (Day 1)

Create `person_b/index.py`. Initialize two persistent Chroma collections — one for flat, one for hierarchy — so they can be loaded without re-embedding:

```python
import chromadb
client = chromadb.PersistentClient(path="data/indexes/")
flat_col      = client.get_or_create_collection("flat_corpus")
hierarchy_col = client.get_or_create_collection("hierarchy_corpus")
```

Test that the persistence works: add 3 dummy documents, close Python, reopen, and confirm they're still there.

### Step 2 — Set up the embedding model (Day 1)

Choose one of:
- `BAAI/bge-large-en-v1.5` — free, strong on legal text, runs locally via sentence-transformers
- `text-embedding-3-small` — OpenAI API, faster to set up, costs ~$0.02 per 1M tokens

Load it in `person_b/embedder.py` and write a function `embed(texts: list[str]) -> list[list[float]]`. Test on 5 sample sentences and confirm the output shape.

While waiting for Person A to finish the real corpus (days 1–3), use a small sample: copy 50 entries from a few raw JSON files and run Person A's parser manually. This lets you test the embedding pipeline without blocking.

### Step 3 — Build the indexing pipeline (Days 2–3)

Write `person_b/build_index.py`. This reads a `.jsonl` file and upserts into the appropriate Chroma collection in batches:

```python
def index_corpus(jsonl_path, collection, batch_size=256):
    chunks = load_jsonl(jsonl_path)
    for batch in batched(chunks, batch_size):
        ids        = [c["chunk_id"] for c in batch]
        texts      = [c["text"]     for c in batch]
        metadatas  = [c["metadata"] for c in batch]
        embeddings = embedder.embed(texts)
        collection.upsert(ids=ids, embeddings=embeddings,
                          documents=texts, metadatas=metadatas)
```

Batch size of 256 is a good starting point. Log progress with `tqdm`. Embedding the full corpus will take time — run it overnight or on a GPU if available.

### Step 4 — Implement dense retrieval (Day 3)

Write `person_b/retrieval.py`. Dense retrieval is a single Chroma query:

```python
def dense_retrieve(query, collection, k=5):
    q_emb = embedder.embed([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return format_results(results)
```

Test this with 5 real tax questions. Check that the returned sections are plausibly relevant.

### Step 5 — Implement BM25 sparse retrieval (Days 3–4)

BM25 operates on raw tokens, not embeddings. Load the full corpus into memory and build a BM25 index:

```python
from rank_bm25 import BM25Okapi

def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks

def bm25_retrieve(query, bm25_index, chunks, k=5):
    scores = bm25_index.get_scores(query.lower().split())
    top_k  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_k]
```

The BM25 index does not persist — you rebuild it from the `.jsonl` on each startup. This takes ~5 seconds for 20k chunks, which is acceptable.

### Step 6 — Implement Reciprocal Rank Fusion (Day 4)

RRF combines dense and sparse rankings. It is about 15 lines:

```python
def rrf(rankings: list[list[str]], k=60) -> list[str]:
    """
    rankings: list of ordered lists of chunk_ids (dense first, bm25 second)
    k: RRF constant (60 is standard)
    """
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_retrieve(query, collection, bm25_idx, chunks, k=5):
    dense_ids = [r["chunk_id"] for r in dense_retrieve(query, collection, k=20)]
    bm25_ids  = [r["chunk_id"] for r in bm25_retrieve(query, bm25_idx, chunks, k=20)]
    fused_ids = rrf([dense_ids, bm25_ids])[:k]
    # look up full chunk objects by ID
    id_map = {c["chunk_id"]: c for c in chunks}
    return [id_map[i] for i in fused_ids if i in id_map]
```

Retrieve top-20 from each method, fuse, return top-5. The over-retrieval before fusion is intentional.

### Step 7 — Expose a clean retrieval interface (Day 4–5)

Write the final `person_b/retrieve.py` with a single entry point that Person C will call:

```python
def retrieve(query: str,
             mode: str = "hybrid",          # "dense" | "bm25" | "hybrid"
             corpus: str = "hierarchy",     # "flat" | "hierarchy"
             k: int = 5) -> list[dict]:
    """
    Returns a list of k chunk dicts from the agreed schema.
    This is the only function Person C needs to import.
    """
```

Write a brief `person_b/README.md` explaining how to call it. This is Person C's API.

### Step 8 — Index the real corpora (Day 8, integration week)

When Person A delivers `flat_corpus.jsonl` and `hierarchy_corpus.jsonl` on day 7, run `build_index.py` over both. This is the first integration step. Verify:

- Chunk counts match Person A's summary stats
- A sample query returns relevant results from both corpora
- The BM25 index loads without errors

---

## Phase 1C: RAG Pipeline (Person C, Days 1–8)

### Step 1 — Design the system prompt (Day 1)

This is the most important thing Person C does. A good system prompt for tax RAG should instruct the model to:

1. Cite the exact section number (e.g., "Under §1381(a)(2)...") for every claim
2. Distinguish clearly between federal (Title 26) and Nevada state provisions when both appear in context
3. Explicitly say "I cannot find a relevant provision in the provided context" rather than hallucinating
4. Never give definitive legal advice — frame everything as "the code states..." not "you should..."

Write this as `person_c/prompts.py` with a `SYSTEM_PROMPT` constant. Get feedback from the group on day 1 or 2.

### Step 2 — Set up the LLM API (Day 1)

Create `person_c/llm.py`. Use either Claude (`anthropic` library) or GPT-4o-mini (`openai` library). Abstract it so Person D can swap models for comparison if needed:

```python
def call_llm(system_prompt: str, user_message: str, model: str = "gpt-4o-mini") -> str:
    # returns the text response
```

Store API keys in `.env`, never in the repo. Test with a single "What is §1381?" query and a hardcoded fake context.

### Step 3 — Build the context formatter (Days 2–3)

The context formatter takes the list of retrieved chunks from Person B and formats them into the prompt. This step matters more than most people expect.

```python
def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into a structured context block.
    Include section number, breadcrumb, and text for each.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {i}]")
        lines.append(f"Section: {chunk['section']}")
        lines.append(f"Path: {chunk['breadcrumb']}")
        lines.append(f"Text: {chunk['text']}")
        lines.append("")
    return "\n".join(lines)
```

Also build the user message template:

```python
USER_TEMPLATE = """
Using only the tax code provisions below, answer the following question.
Cite the specific section number for each point you make.

--- TAX CODE CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
"""
```

### Step 4 — Build the core RAG function (Day 3)

Wire retrieval + formatting + generation into one function:

```python
def answer(question: str,
           retrieval_mode: str = "hybrid",
           corpus: str = "hierarchy") -> dict:
    chunks   = retrieve(question, mode=retrieval_mode, corpus=corpus)
    context  = format_context(chunks)
    prompt   = USER_TEMPLATE.format(context=context, question=question)
    response = call_llm(SYSTEM_PROMPT, prompt)
    return {
        "question":   question,
        "answer":     response,
        "sources":    chunks,
        "config":     {"retrieval_mode": retrieval_mode, "corpus": corpus}
    }
```

At this stage, use Person B's sample index (not the full real corpus). The function should be callable and return real output by end of day 3.

### Step 5 — Handle the no-RAG baseline config (Day 4)

Add a `no_rag` mode to the `answer()` function that skips retrieval entirely and asks the LLM the question with no context. This is one of the four configurations Person D will evaluate. It just calls the LLM with an empty context section.

### Step 6 — Build the Streamlit UI (Days 5–7)

Create `person_c/app.py`. Keep it simple — the evaluation is what matters, not the UI polish:

```python
import streamlit as st

st.title("Tax Code RAG")

corpus_mode    = st.radio("Corpus", ["flat", "hierarchy"])
retrieval_mode = st.radio("Retrieval", ["dense", "bm25", "hybrid"])
no_rag_mode    = st.checkbox("No-RAG baseline (ignore retrieval)")
question       = st.text_area("Ask a tax question")

if st.button("Submit"):
    result = answer(question, ...)
    st.write(result["answer"])
    with st.expander("Sources"):
        for src in result["sources"]:
            st.caption(f"{src['section']} — {src['breadcrumb']}")
            st.write(src['text'])
```

Run with `streamlit run person_c/app.py`. This is what you demo at the end.

### Step 7 — Connect to real indexes (Days 8–9, integration)

When Person B finishes indexing the real corpora on day 8, update the `retrieve()` call to point at the real Chroma collections. Run 10 manual test questions through all four configurations and visually confirm outputs look reasonable. Flag any obvious failures to the group.

---

## Phase 1D: Evaluation (Person D, Days 1–7)

### Step 1 — Load and inspect the HuggingFace dataset (Day 1)

```python
from datasets import load_dataset
ds = load_dataset("quotientai/irs_form_instruction_qa_pairs")
print(ds)
print(ds["train"][0])  # inspect one sample
```

Check:
- What fields does each record have? (question, answer, source document, etc.)
- How many QA pairs are there?
- Are the answers short (extractive) or long (generative)?
- Do the questions reference specific section numbers? These are the easiest to evaluate.

Write your observations in `person_d/NOTES.md`. Select a test subset: take 100–200 questions that have clear, verifiable answers. Store them as `data/eval_set.jsonl`.

### Step 2 — Write the retrieval evaluator (Days 2–3)

For retrieval evaluation you need to know which chunk(s) contain the correct answer. Since you may not have ground-truth chunk IDs, use a proxy: check whether the correct section number appears in the top-k retrieved chunks.

```python
def retrieval_recall_at_k(question, expected_section, retriever_fn, k=5):
    """
    Returns 1 if expected_section appears in top-k results, else 0.
    expected_section: e.g. "§1381" or "§162"
    """
    chunks = retriever_fn(question, k=k)
    retrieved_sections = [c["section"] for c in chunks]
    return int(expected_section in retrieved_sections)
```

Also implement MRR (Mean Reciprocal Rank):

```python
def mrr(question, expected_section, retriever_fn, k=10):
    chunks = retriever_fn(question, k=k)
    for rank, chunk in enumerate(chunks, 1):
        if chunk["section"] == expected_section:
            return 1 / rank
    return 0.0
```

You can extract `expected_section` from the HuggingFace dataset records — most IRS QA pairs reference a specific form or code section.

### Step 3 — Write the generation evaluator (Days 3–5)

For generation quality, use LLM-as-judge. This is the most informative metric for open-ended answers:

```python
JUDGE_PROMPT = """
You are evaluating a tax question answering system.

Question: {question}
Reference answer: {reference}
System answer: {prediction}

Score the system answer on two dimensions, each 1–5:
1. Faithfulness: does the answer stay within what the tax code says? (1=hallucinated, 5=fully grounded)
2. Completeness: does it address all key points in the reference? (1=missing most, 5=covers all)

Respond ONLY with JSON: {{"faithfulness": N, "completeness": N}}
"""

def llm_judge(question, reference, prediction):
    prompt = JUDGE_PROMPT.format(...)
    response = call_llm("You are a precise evaluator.", prompt)
    return json.loads(response)  # {"faithfulness": 4, "completeness": 3}
```

Run this on a sample of 20 questions manually first and read the scores. If they seem off, adjust the prompt before running the full eval.

### Step 4 — Build the eval harness (Days 5–6)

Write `person_d/run_eval.py`. This is the main script that runs all four system configurations against the test set and saves results:

```python
CONFIGS = [
    {"name": "no_rag",             "corpus": None,        "mode": "none"},
    {"name": "flat_dense",         "corpus": "flat",      "mode": "dense"},
    {"name": "hierarchy_hybrid",   "corpus": "hierarchy", "mode": "hybrid"},
    {"name": "hierarchy_dense",    "corpus": "hierarchy", "mode": "dense"},
]

results = []
for q in tqdm(eval_set):
    row = {"question": q["question"], "reference": q["answer"]}
    for cfg in CONFIGS:
        output = answer(q["question"], **cfg)
        row[f"{cfg['name']}_answer"]       = output["answer"]
        row[f"{cfg['name']}_recall"]       = retrieval_recall_at_k(...)
        row[f"{cfg['name']}_faithfulness"] = llm_judge(...)["faithfulness"]
        row[f"{cfg['name']}_completeness"] = llm_judge(...)["completeness"]
    results.append(row)

pd.DataFrame(results).to_csv("outputs/eval_results/full_eval.csv")
```

Important: add a `--dry-run` flag that runs on 5 questions only. Use this to test before the full run, which will take 30–60 minutes and cost API money.

### Step 5 — Build the comparison table (Day 7)

Write `person_d/analyze.py` that loads `full_eval.csv` and produces summary stats:

```python
for cfg in CONFIGS:
    print(f"\n{cfg['name']}")
    print(f"  Recall@5:       {df[f'{cfg}_recall'].mean():.3f}")
    print(f"  Faithfulness:   {df[f'{cfg}_faithfulness'].mean():.2f} / 5")
    print(f"  Completeness:   {df[f'{cfg}_completeness'].mean():.2f} / 5")
```

Also write the 10 questions where hierarchy RAG beat flat RAG the most, and the 10 where it didn't help. These are your qualitative findings.

---

## Phase 2: Integration (Days 8–10, all four)

These steps happen in strict order.

**Step 2.1 (Day 8) — A hands off corpora to B**
Person A announces in the group chat that both `.jsonl` files are in `data/chunks/`. Person B pulls the repo, runs `build_index.py` over both files. Expected time: 2–4 hours depending on hardware. Person A stays available to fix any parsing bugs that surface at scale.

**Step 2.2 (Day 9) — B hands retrieval module to C**
Person B announces that both Chroma indexes are built and the `retrieve()` function is ready. Schedule a 30-minute sync call. Person C imports `from person_b.retrieve import retrieve`, runs a test query, confirms results look sensible. If the Chroma path config is wrong, fix it together on the call.

**Step 2.3 (Day 9–10) — C and D align on the answer() interface**
Person D's eval harness calls `answer()` from Person C's module. Confirm the function signature matches what D's harness expects. Run the `--dry-run` eval (5 questions, all 4 configs) together. Fix any import errors or config mismatches.

**Step 2.4 (Day 10) — Smoke test the full stack**
All four open the Streamlit app. Each person asks 3 questions they personally find interesting. Check:
- Does no-RAG hallucinate section numbers? (It should.)
- Does hierarchy RAG cite breadcrumbs correctly?
- Does hybrid retrieval pull different results than dense-only?

If something breaks, fix it before running the full eval.

---

## Phase 3: Evaluation Run (Days 11–13)

**Step 3.1 (Day 11) — Run full eval suite**
Person D runs `run_eval.py` (no dry-run flag) on the full test set across all 4 configs. This runs overnight or takes a few hours. Monitor the first 10 rows to confirm nothing is crashing silently.

**Step 3.2 (Day 12) — Inspect failure cases**
Person D runs `analyze.py` and shares the summary table with the group. Everyone looks at specific failure cases:
- Where does no-RAG fail? (Wrong section numbers? Fabricated rules?)
- Where does flat RAG fail but hierarchy succeeds? (Nested subsections?)
- Where does hierarchy RAG fail despite good retrieval? (LLM misreads context?)

**Step 3.3 (Day 13) — Write the findings**
Each person writes one paragraph:
- A: notes on what the chunking strategy affected
- B: notes on what hybrid retrieval added vs dense-only
- C: notes on prompt design choices and their effect
- D: the comparison table + overall interpretation

Combine into `outputs/final_report.md`.

---

## Checklist: Definition of Done

- [ ] `flat_corpus.jsonl` and `hierarchy_corpus.jsonl` exist and pass the quality checks
- [ ] Both Chroma indexes are built and queryable
- [ ] `retrieve(query, mode, corpus, k)` works for all 3 modes × 2 corpora
- [ ] `answer(question, retrieval_mode, corpus)` returns citation-grounded answers
- [ ] Streamlit app runs locally and shows sources
- [ ] `full_eval.csv` exists with scores for all 4 configurations
- [ ] Comparison table shows statistically distinguishable results between configs
- [ ] `final_report.md` written and committed to main
