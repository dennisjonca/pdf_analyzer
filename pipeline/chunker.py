from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG
from utils.logger import get_logger


logger = get_logger("chunker")


@dataclass
class ChunkMeta:
    quelldatei: str
    seite: int
    doc_type: str
    tabellen_typ: str | None
    extraktions_methode: str = "digital"


@dataclass
class ChunkResult:
    text: str
    metadata: dict


class Chunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or CONFIG["chunk_overlap"]

    def chunk_tabelle(self, dataframe, meta: ChunkMeta) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        for index, (_, row) in enumerate(dataframe.fillna("").iterrows(), start=1):
            row_payload = " | ".join(
                f"{column}: {row[column]}" for column in dataframe.columns if str(row[column]).strip()
            )
            if not row_payload:
                continue
            text = (
                f"Dokumenttyp: {meta.doc_type} | Tabellentyp: {meta.tabellen_typ or 'unbekannt'}\n"
                f"Quelle: {meta.quelldatei}, Seite {meta.seite}, Zeile {index}\n"
                f"{row_payload}"
            )
            metadata = {
                "quelldatei": meta.quelldatei,
                "seite": meta.seite,
                "doc_type": meta.doc_type,
                "tabellen_typ": meta.tabellen_typ,
                "chunk_typ": "tabelle",
                "extraktions_methode": meta.extraktions_methode,
            }
            for key in ("datum", "faelligkeitsdatum", "gesamtbetrag", "rechnungsnummer", "lieferant", "kunde"):
                if key in dataframe.columns and row.get(key) not in ("", None):
                    metadata[key] = row.get(key)
            results.append(ChunkResult(text=text, metadata=metadata))
        return results

    def chunk_fliesstext(self, text: str, meta: ChunkMeta) -> list[ChunkResult]:
        if not text.strip():
            return []
        chunks = self._split_text(text)
        results = []
        for chunk in chunks:
            prefixed_chunk = (
                f"Dokumenttyp: {meta.doc_type}\nQuelle: {meta.quelldatei}, Seite {meta.seite}\n{chunk}"
            )
            results.append(
                ChunkResult(
                    text=prefixed_chunk,
                    metadata={
                        "quelldatei": meta.quelldatei,
                        "seite": meta.seite,
                        "doc_type": meta.doc_type,
                        "tabellen_typ": meta.tabellen_typ,
                        "chunk_typ": "fliesstext",
                        "extraktions_methode": meta.extraktions_methode,
                    },
                )
            )
        return results

    def _split_text(self, text: str) -> list[str]:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return splitter.split_text(text)
        except Exception:
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                if end >= len(text):
                    break
                start = max(end - self.chunk_overlap, start + 1)
            return chunks
