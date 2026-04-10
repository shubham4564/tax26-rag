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
