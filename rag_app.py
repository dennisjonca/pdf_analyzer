#!/usr/bin/env python3
"""Local PDF RAG with Ollama + ChromaDB + Hybrid Search + Streamlit.

CLI examples:
  python rag_app.py index --pdf-dir ./pdfs
  python rag_app.py ask --question "What is this document about?"

Streamlit UI:
  streamlit run rag_app.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pdf_chunks")


def pdf_documents(pdf_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for p in pages:
            p.metadata = {
                **p.metadata,
                "source": str(pdf),
                "filename": pdf.name,
                "page": p.metadata.get("page", 0),
            }
        docs.extend(pages)
    return docs


def split_documents(docs: Sequence[Document], chunk_size: int = 1000, overlap: int = 150) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(list(docs))


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def index_pdfs(pdf_dir: Path) -> Tuple[int, int]:
    docs = pdf_documents(pdf_dir)
    chunks = split_documents(docs)
    store = get_vectorstore()
    if chunks:
        store.add_documents(chunks)
    return len(docs), len(chunks)


def build_hybrid_retriever(k: int = 8) -> EnsembleRetriever:
    store = get_vectorstore()
    if store._collection.count() == 0:  # pylint: disable=protected-access
        raise ValueError(
            "No indexed documents found. Run `python rag_app.py index --pdf-dir ./pdfs` first."
        )
    vector_retriever = store.as_retriever(search_kwargs={"k": k})

    all_docs = store.similarity_search(" ", k=300)
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k

    return EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.4, 0.6])


def rerank(question: str, docs: Sequence[Document], top_n: int = 5) -> List[Document]:
    if not docs:
        return []
    compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANK_MODEL), top_n=top_n)
    return compressor.compress_documents(list(docs), query=question)


def answer_question(question: str, k: int = 8, top_n: int = 5) -> Tuple[str, List[Document]]:
    retriever = build_hybrid_retriever(k=k)
    retrieved = retriever.invoke(question)
    final_docs = rerank(question, retrieved, top_n=top_n)

    context = "\n\n".join(
        f"[Source: {d.metadata.get('filename', 'unknown')} | Page: {d.metadata.get('page', 'n/a')}]\n{d.page_content}"
        for d in final_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful PDF analyst. Only answer from the provided context. "
                "If context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\n"
                "Provide a concise answer and cite sources as [filename p.X].",
            ),
        ]
    )

    chain = prompt | ChatOllama(model=LLM_MODEL, temperature=0) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer, final_docs


def render_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Local PDF RAG Analyzer", layout="wide")
    st.title("📄 Local PDF RAG Analyzer")
    st.caption("Ollama + ChromaDB + Hybrid Search (BM25 + Vector) + Re-ranking")

    pdf_dir = st.text_input("PDF directory", value="./pdfs")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index PDFs"):
            docs_n, chunks_n = index_pdfs(Path(pdf_dir))
            st.success(f"Indexed {docs_n} pages into {chunks_n} chunks.")

    question = st.text_area("Ask a question about your PDFs")
    if st.button("Run RAG") and question.strip():
        with st.spinner("Thinking..."):
            answer, docs = answer_question(question)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source citations")
        for i, d in enumerate(docs, start=1):
            st.markdown(
                f"**{i}. {d.metadata.get('filename', 'unknown')} (p.{d.metadata.get('page', 'n/a')})**"
            )
            st.code(d.page_content[:1000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PDF RAG app")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Index PDFs into ChromaDB")
    p_index.add_argument("--pdf-dir", default="./pdfs", type=Path)

    p_ask = sub.add_parser("ask", help="Ask question over indexed PDFs")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--k", type=int, default=8)
    p_ask.add_argument("--top-n", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "index":
        docs_n, chunks_n = index_pdfs(args.pdf_dir)
        print(f"Indexed {docs_n} pages into {chunks_n} chunks in {CHROMA_DIR}")
    elif args.cmd == "ask":
        try:
            answer, docs = answer_question(args.question, k=args.k, top_n=args.top_n)
        except ValueError as exc:
            print(str(exc))
            return
        print("\nAnswer:\n")
        print(answer)
        print("\nCitations:")
        for d in docs:
            print(f"- {d.metadata.get('filename', 'unknown')} p.{d.metadata.get('page', 'n/a')}")
    else:
        render_streamlit()


if __name__ == "__main__":
    main()
