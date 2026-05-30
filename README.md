# pdf_analyzer
Using free Ollama to read pdf files. This repository serves solely as a learning resource. It utilizes specific tools and breaks them down into small, manageable topics, thereby facilitating a better understanding of the individual steps involved in similar systems.

## Complete local RAG example (Ollama + ChromaDB + LangChain)

This repository now includes a complete, runnable Python script at:

- `/tmp/workspace/dennisjonca/pdf_analyzer/rag_app.py`

Features included:
- Local Ollama LLM + real embedding model (`nomic-embed-text` by default)
- PDF ingestion with metadata (`filename`, `source`, `page`)
- ChromaDB persistent vector store
- Hybrid retrieval (BM25 + Vector Search)
- Re-ranking (`BAAI/bge-reranker-base` by default)
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
ollama pull nomic-embed-text
```

Create a `pdfs/` folder and add your PDF files.

## CLI usage

Index PDFs:

```bash
python rag_app.py index --pdf-dir ./pdfs
```

> Note: for safety, `--pdf-dir` must resolve to a directory inside the current working directory.

Ask a question:

```bash
python rag_app.py ask --question "What are the main conclusions?"
```

## Streamlit usage

```bash
streamlit run rag_app.py
```

## Optional environment variables

- `LLM_MODEL` (default: `llama3.1:8b`)
- `EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `RERANK_MODEL` (default: `BAAI/bge-reranker-base`)
- `CHROMA_DIR` (default: `./chroma_db`)
- `CHROMA_COLLECTION` (default: `pdf_chunks`)
