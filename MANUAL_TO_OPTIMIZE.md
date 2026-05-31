# Manual to Optimize

This guide is focused on technical PDFs with tables, lists, product names, numbers, specs, and manufacturers.

## 1) Key components and how they relate

- **Chunking** defines what each searchable unit contains.
- **Embedding model** converts chunks and query into vectors (semantic matching).
- **BM25** scores exact words/tokens (great for model names, numbers, part IDs).
- **Hybrid retrieval** combines vector + BM25 (broad meaning + exact technical terms).
- **Re-ranking** reorders retrieved chunks using a stronger model before answer generation.
- **LLM generation** uses top re-ranked chunks to produce final answer/CSV.

If chunking is weak, every later stage is weaker. Start tuning chunking first.

## 2) Chunking parameters (most important first)

### `CHUNK_SIZE` / `index --chunk-size`
- Larger chunks = more context, but can dilute precision.
- Smaller chunks = higher precision, but risk missing context.
- Good starting range for technical PDFs: **500–900**.

### `CHUNK_OVERLAP` / `index --chunk-overlap`
- Helps preserve context across chunk boundaries.
- Typical range: **80–220**.
- Keep overlap clearly smaller than chunk size.

### `TABLE_LINES_PER_CHUNK`
- For table/list-like blocks, lines are grouped before recursive splitting.
- Lower value = more precise rows; higher value = more context around each row.
- Typical range: **4–10**.

## 3) Retrieval and re-ranking parameters

### `RETRIEVAL_K`
- Number of candidates fetched before re-ranking.
- Increase when answers miss relevant specs/manufacturer details.
- Typical range: **12–24** for technical corpora.

### `RERANK_TOP_N`
- Number of final chunks passed to generation.
- Too low can miss evidence; too high can add noise.
- Typical range: **4–10**.

## 4) Model choices

### `EMBEDDING_MODEL`
- Default: `mxbai-embed-large` (good quality for retrieval).
- If semantic matches are weak, test alternative embedding models.

### `RERANK_MODEL`
- Default: `BAAI/bge-reranker-v2-m3`.
- Strong rerankers help with technical comparisons and exact field extraction.

### `LLM_MODEL`
- Must reliably follow format instructions (especially CSV output).
- For extraction-heavy tasks, prefer deterministic settings (temperature already fixed to 0).

## 5) Practical tuning loop

1. Index once with baseline:
   - `python rag_app.py index --pdf-dir ./pdfs --chunk-size 700 --chunk-overlap 160`
2. Run 5–10 representative technical questions (names/specs/manufacturer queries).
3. If misses exact values: increase `RETRIEVAL_K`, reduce `TABLE_LINES_PER_CHUNK`.
4. If answers are noisy: reduce `RETRIEVAL_K` or increase `RERANK_TOP_N` quality balance.
5. Re-index after changing chunking params.

## 6) Example profiles

### High precision extraction (tables/spec rows)
- `CHUNK_SIZE=550`
- `CHUNK_OVERLAP=120`
- `TABLE_LINES_PER_CHUNK=4`
- `RETRIEVAL_K=18`
- `RERANK_TOP_N=6`

### Balanced technical QA
- `CHUNK_SIZE=700`
- `CHUNK_OVERLAP=160`
- `TABLE_LINES_PER_CHUNK=6`
- `RETRIEVAL_K=12`
- `RERANK_TOP_N=6`

### Broader contextual answers
- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=220`
- `TABLE_LINES_PER_CHUNK=8`
- `RETRIEVAL_K=16`
- `RERANK_TOP_N=8`
