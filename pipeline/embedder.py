from __future__ import annotations

import shutil
from pathlib import Path

from config import CONFIG
from utils.file_utils import ensure_directory
from utils.logger import get_logger


logger = get_logger("embedder")


class ChromaEmbedder:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or CONFIG
        ensure_directory(self.config["chroma_dir"])
        self._vectorstore = None

    def reset(self) -> None:
        chroma_dir = Path(self.config["chroma_dir"])
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = None

    def add_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0
        vectorstore = self._get_vectorstore()
        total = 0
        batch_size = self.config["embedding_batch_size"]
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            metadatas = [self._sanitize_metadata(chunk.metadata) for chunk in batch]
            ids = [self._chunk_id(metadata, start + offset) for offset, metadata in enumerate(metadatas)]
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            total += len(batch)
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
        logger.info("%s Vektoren gespeichert in ChromaDB", total)
        return total

    def similarity_search(self, query: str, top_k: int, metadata_filter: dict | None = None):
        vectorstore = self._get_vectorstore()
        try:
            return vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
                filter=metadata_filter,
            )
        except TypeError:
            docs = vectorstore.similarity_search(query=query, k=top_k, filter=metadata_filter)
            return [(doc, None) for doc in docs]

    def _get_vectorstore(self):
        if self._vectorstore is None:
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                from langchain_community.vectorstores import Chroma
            except ImportError as exc:
                raise RuntimeError(
                    "LangChain Community, ChromaDB und Ollama sind für Embeddings erforderlich."
                ) from exc
            embeddings = OllamaEmbeddings(model=self.config["embed_model"])
            self._vectorstore = Chroma(
                collection_name=self.config["collection_name"],
                embedding_function=embeddings,
                persist_directory=self.config["chroma_dir"],
            )
        return self._vectorstore

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (list, dict, tuple, set)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def _chunk_id(metadata: dict, index: int) -> str:
        source = metadata.get("quelldatei", "unbekannt")
        page = metadata.get("seite", 0)
        chunk_type = metadata.get("chunk_typ", "chunk")
        return f"{source}:{page}:{chunk_type}:{index}"
