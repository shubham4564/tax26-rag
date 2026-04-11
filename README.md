# tax26-rag

This is our course project for tax-law question answering with RAG.

The idea is simple: we ingest federal + Nevada tax text, build two retrieval views (flat and hierarchy-aware), and then compare how much retrieval setup changes answer quality.

The code is organized by workflow stage:

- `ingestion/` builds corpora from raw data
- `retrieval/` builds indexes and serves chunk retrieval
- `generation/` handles prompts + LLM calls + answer pipeline
- `evaluation/` runs dataset-based evaluation and summary reports
- `app.py` is the Streamlit demo

## Data Used

This project uses two legal sources for retrieval:

- Federal tax law (Title 26 / IRC), parsed from JSON section files.
- Nevada tax law files, parsed into the same chunk format.

Main data locations:

- Raw input root: `data/raw/`
- Federal sections consumed by ingestion: `data/raw/federal/usc_title26_olrc/sections/`
- Nevada raw files expected by ingestion: `data/raw/nevada/`
- Generated corpora: `data/chunks/flat_corpus.jsonl` and `data/chunks/hierarchy_corpus.jsonl`
- Vector index storage (Chroma): `data/indexes/`
- Evaluation set (generated from Hugging Face in `evaluation.load_dataset`): `data/eval_set.jsonl`
- Evaluation outputs and report: `outputs/eval_results/`

If you place data in different folders, update paths in `config.py` (and, if needed, `ingestion/run_pipeline.py`).

## Quick Start

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

Use one of these examples (pick one provider at a time):

Gemini

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-flash
```

OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
```

Anthropic

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Perplexity

```env
LLM_PROVIDER=perplexity
PERPLEXITY_API_KEY=your_perplexity_key_here
PERPLEXITY_MODEL=sonar
```

## Typical Run Order

Run these from the project root.

### 1) Build corpora

```bash
python -m ingestion.run_pipeline
```

Outputs:
- `data/chunks/flat_corpus.jsonl`
- `data/chunks/hierarchy_corpus.jsonl`

### 2) Build retrieval indexes

```bash
python -m retrieval.build_index
```

Rebuild from scratch when needed:

```bash
python -m retrieval.build_index --rebuild
```

Quick retrieval smoke test:

```bash
python -m retrieval.retrieve
```

### 3) Run generation pipeline and app

Quick pipeline test:

```bash
python -m generation.pipeline
```

Launch UI:

```bash
streamlit run app.py
```

### 4) Evaluate

Prepare evaluation set:

```bash
python -m evaluation.load_dataset
```

Dry run (5 samples):

```bash
python -m evaluation.run_eval --dry-run
```

Full run:

```bash
python -m evaluation.run_eval
```

Analyze results + report:

```bash
python -m evaluation.analyze
```

## Notes

- If API calls fail with 429, you are likely hitting quota/rate limits.
- Keep your `.env` local and never commit API keys.
- If imports look stale, make sure you are using the new package names (`ingestion`, `retrieval`, `generation`, `evaluation`).
