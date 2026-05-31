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
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import chromadb
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pdf_chunks")
APP_LANGUAGE = os.getenv("APP_LANGUAGE", "en")
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "12"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))
LANGUAGE_NAMES = {"en": "English", "de": "German"}
_RERANKER: CrossEncoder | None = None


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class RankBM25Retriever(BaseRetriever):
    docs: List[Document]
    k: int = DEFAULT_RETRIEVAL_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not self.docs:
            return []
        tokenized_corpus = [_tokenize_for_bm25(doc.page_content) for doc in self.docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize_for_bm25(query))
        top_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[: self.k]
        return [self.docs[i] for i in top_indices]


def validate_pdf_dir(pdf_dir: Path) -> Path:
    candidate = pdf_dir.expanduser().resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError(f"Not a directory: {candidate}")

    base_dir = Path.cwd().resolve()
    if candidate != base_dir and base_dir not in candidate.parents:
        raise ValueError(
            f"Directory must be inside {base_dir}. Received: {candidate}"
        )
    return candidate


def pdf_documents(pdf_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for page_index, page in enumerate(reader.pages):
            docs.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(pdf), "filename": pdf.name, "page": page_index},
                )
            )
    return docs


def normalize_language(language: str) -> str:
    value = language.strip().lower()
    if value not in LANGUAGE_NAMES:
        raise ValueError("Unsupported language. Use one of: en, de.")
    return value


def is_csv_request(question: str) -> bool:
    value = question.lower()
    keywords = ("csv", "comma-separated", "comma separated", "kommagetrennt")
    return any(keyword in value for keyword in keywords)


def is_heading(text: str) -> bool:
    value = text.strip()
    if not value or len(value) > 120:
        return False
    if value.endswith((".", "!", "?", ";", ":")):
        return False
    words = value.split()
    if len(words) > 12:
        return False
    if re.match(r"^(\d+(\.\d+)*)\s+[A-ZÄÖÜ]", value):
        return True
    uppercase_ratio = sum(1 for c in value if c.isupper()) / max(1, sum(1 for c in value if c.isalpha()))
    return uppercase_ratio > 0.6 or value.istitle()


def paragraph_documents(docs: Sequence[Document]) -> List[Document]:
    paragraph_docs: List[Document] = []
    for doc in docs:
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", doc.page_content.replace("\r\n", "\n"))
            if part.strip()
        ]
        current_heading = doc.metadata.get("heading")
        for part in paragraphs:
            if is_heading(part):
                current_heading = part
                continue
            metadata = dict(doc.metadata)
            if current_heading:
                metadata["heading"] = current_heading
            paragraph_docs.append(Document(page_content=part, metadata=metadata))
    return paragraph_docs or list(docs)


def split_documents(docs: Sequence[Document], chunk_size: int = 1000, overlap: int = 150) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(paragraph_documents(docs))


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
    docs = pdf_documents(validate_pdf_dir(pdf_dir))
    chunks = split_documents(docs)
    store = get_vectorstore()
    store.delete_collection()
    store = get_vectorstore()
    if chunks:
        store.add_documents(chunks)
    return len(docs), len(chunks)


def build_hybrid_retriever(k: int = DEFAULT_RETRIEVAL_K) -> EnsembleRetriever:
    store = get_vectorstore()
    if store._collection.count() == 0:  # pylint: disable=protected-access
        raise ValueError(
            "No indexed documents found. Run `python rag_app.py index --pdf-dir ./pdfs` first."
        )
    vector_retriever = store.as_retriever(search_kwargs={"k": k})

    all_items = store._collection.get(include=["documents", "metadatas"])  # pylint: disable=protected-access
    all_docs = [
        Document(page_content=page_content, metadata=metadata or {})
        for page_content, metadata in zip(all_items.get("documents", []), all_items.get("metadatas", []))
    ]
    if not all_docs:
        all_docs = store.similarity_search(" ", k=max(k * 20, 300))
    bm25 = RankBM25Retriever(docs=all_docs, k=k)

    return EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.4, 0.6])


def rerank(question: str, docs: Sequence[Document], top_n: int = DEFAULT_RERANK_TOP_N) -> List[Document]:
    if not docs:
        return []
    docs_list = list(docs)
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANK_MODEL)
    pairs = [(question, doc.page_content) for doc in docs_list]
    scores = _RERANKER.predict(pairs)
    top_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_n]
    return [docs_list[i] for i in top_indices]


def answer_question(
    question: str,
    k: int = DEFAULT_RETRIEVAL_K,
    top_n: int = DEFAULT_RERANK_TOP_N,
    language: str = "en",
    output_format: str = "text",
) -> Tuple[str, List[Document]]:
    language = normalize_language(language)
    language_name = LANGUAGE_NAMES[language]
    if output_format not in {"text", "csv"}:
        raise ValueError("Unsupported output format. Use 'text' or 'csv'.")
    retriever = build_hybrid_retriever(k=k)
    retrieved = retriever.invoke(question)
    final_docs = rerank(question, retrieved, top_n=top_n)

    context = "\n\n".join(
        f"[Source: {d.metadata.get('filename', 'unknown')} | Page: {d.metadata.get('page', 'n/a')}]\n{d.page_content}"
        for d in final_docs
    )

    format_instruction = (
        "Return only RFC4180-compatible CSV with a single header row. "
        "Do not include markdown code fences or extra explanatory text."
        if output_format == "csv"
        else "Provide a concise answer and cite sources as [filename p.X]."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful PDF analyst. Only answer from the provided context. "
                "If context is insufficient, say so clearly. "
                "Respond in {language_name}.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\n"
                "{format_instruction}",
            ),
        ]
    )

    chain = prompt | ChatOllama(model=LLM_MODEL, temperature=0) | StrOutputParser()
    answer = chain.invoke(
        {
            "question": question,
            "context": context,
            "language_name": language_name,
            "format_instruction": format_instruction,
        }
    )
    return answer, final_docs


def render_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Local PDF RAG Analyzer", layout="wide")
    st.title("📄 Local PDF RAG Analyzer")
    st.caption("Ollama + ChromaDB + Hybrid Search (BM25 + Vector) + Re-ranking")
    language_labels = {"English": "en", "Deutsch": "de"}
    selected_label = st.selectbox("Antwortsprache / Answer language", tuple(language_labels.keys()), index=1)
    answer_language = language_labels[selected_label]

    pdf_dir = st.text_input("PDF directory", value="./pdfs")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index PDFs"):
            docs_n, chunks_n = index_pdfs(Path(pdf_dir))
            st.success(f"Indexed {docs_n} pages into {chunks_n} chunks.")

    question = st.text_area("Ask a question about your PDFs")
    if st.button("Run RAG") and question.strip():
        with st.spinner("Thinking..."):
            csv_requested = is_csv_request(question)
            output_format = "csv" if csv_requested else "text"
            answer, docs = answer_question(question, language=answer_language, output_format=output_format)
        st.subheader("Answer")
        st.write(answer)
        if csv_requested:
            st.download_button(
                "Download CSV",
                data=answer,
                file_name="answer.csv",
                mime="text/csv",
            )

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
    p_ask.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K)
    p_ask.add_argument("--top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    p_ask.add_argument("--language", default=APP_LANGUAGE, choices=tuple(LANGUAGE_NAMES.keys()))
    p_ask.add_argument("--csv-out", default=None, help="Optional path to write CSV output")

    args = parser.parse_args()

    if args.cmd == "index":
        try:
            docs_n, chunks_n = index_pdfs(args.pdf_dir)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc))
            return
        print(f"Indexed {docs_n} pages into {chunks_n} chunks in {CHROMA_DIR}")
    elif args.cmd == "ask":
        try:
            csv_requested = bool(args.csv_out) or is_csv_request(args.question)
            output_format = "csv" if csv_requested else "text"
            answer, docs = answer_question(
                args.question,
                k=args.k,
                top_n=args.top_n,
                language=args.language,
                output_format=output_format,
            )
        except ValueError as exc:
            print(str(exc))
            return
        print("\nAnswer:\n")
        print(answer)
        if csv_requested:
            csv_path = Path(args.csv_out or "answer.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(answer, encoding="utf-8")
            print(f"\nCSV written to: {csv_path}")
        print("\nCitations:")
        for d in docs:
            print(f"- {d.metadata.get('filename', 'unknown')} p.{d.metadata.get('page', 'n/a')}")
    else:
        render_streamlit()


if __name__ == "__main__":
    main()
