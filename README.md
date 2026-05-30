# pdf_analyzer
Using free Ollama to read pdf files. This repository serves solely as a learning resource. It utilizes specific tools and breaks them down into small, manageable topics, thereby facilitating a better understanding of the individual steps involved in similar systems.

## Complete local RAG example (Ollama + ChromaDB + LangChain)

This repository now includes a complete, runnable Python script at:

- `/tmp/workspace/dennisjonca/pdf_analyzer/rag_app.py`

Features included:
- Local Ollama LLM + stronger embedding model (`mxbai-embed-large` by default)
- PDF ingestion with metadata (`filename`, `source`, `page`)
- ChromaDB persistent vector store
- Paragraph-aware, heading-aware chunking with page-preserving metadata
- Hybrid retrieval (BM25 + Vector Search) with larger default candidate set
- Re-ranking (`BAAI/bge-reranker-v2-m3` by default)
- Source citations in responses
- Streamlit web interface

## Setup

```bash
cd /tmp/workspace/dennisjonca/pdf_analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running locally and pull models:

```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

Create a `pdfs/` folder and add your PDF files.

## CLI usage

Index PDFs:

```bash
python rag_app.py index --pdf-dir ./pdfs
```

> Note: for safety, `--pdf-dir` must resolve to a directory inside the current working directory.
> Running `index` refreshes the index for the collection (old indexed PDFs are replaced).

Ask a question:

```bash
python rag_app.py ask --question "What are the main conclusions?"
```

Ask in German (Antwort auf Deutsch):

```bash
python rag_app.py ask --question "Was sind die wichtigsten Erkenntnisse?" --language de
```

Ask for CSV output and save to file:

```bash
python rag_app.py ask --question "List product names and prices as CSV" --csv-out ./output/answer.csv
```

## Streamlit usage

```bash
streamlit run rag_app.py
```

In Streamlit, choose **"Antwortsprache / Answer language"** and select **Deutsch** for German answers.
If your question asks for CSV output, the app also shows a **Download CSV** button.

## Optional environment variables

- `LLM_MODEL` (default: `llama3.1:8b`)
- `EMBEDDING_MODEL` (default: `mxbai-embed-large`)
- `RERANK_MODEL` (default: `BAAI/bge-reranker-v2-m3`)
- `RETRIEVAL_K` (default: `12`)
- `RERANK_TOP_N` (default: `6`)
- `CHROMA_DIR` (default: `./chroma_db`)
- `CHROMA_COLLECTION` (default: `pdf_chunks`)
- `APP_LANGUAGE` (default: `en`, supported: `en`, `de`)
