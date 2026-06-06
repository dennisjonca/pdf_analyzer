from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from config import CONFIG
from pipeline.chunker import ChunkMeta, Chunker
from pipeline.classifier import DocumentClassifier
from pipeline.embedder import ChromaEmbedder
from pipeline.extractor import PDFExtractor
from pipeline.normalizer import SemanticNormalizer
from pipeline.table_detector import TableTypeDetector
from utils.error_queue import ErrorQueue
from utils.file_utils import (
    ensure_directory,
    file_hash,
    load_json,
    scan_pdfs,
    write_json,
)
from utils.logger import get_logger


logger = get_logger("batch_processor")


@dataclass
class ProcessingResult:
    pdf_path: str
    success: bool
    doc_type: str
    chunks_created: int
    tables_found: int
    error_message: str | None
    processing_time_s: float


@dataclass
class BatchReport:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[ProcessingResult] = field(default_factory=list)


class BatchProcessor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or CONFIG
        ensure_directory(self.config["input_dir"])
        ensure_directory(self.config["log_dir"])
        ensure_directory(self.config["error_dir"])
        self.classifier = DocumentClassifier()
        self.extractor = PDFExtractor(self.config)
        self.table_detector = TableTypeDetector(self.config)
        self.normalizer = SemanticNormalizer()
        self.chunker = Chunker()
        self.embedder = ChromaEmbedder(self.config)
        self.error_queue = ErrorQueue(self.config["error_dir"])

    def run(self, input_dir: str) -> BatchReport:
        report = BatchReport()
        processed_hashes = self.get_processed_hashes()
        pdf_paths = scan_pdfs(input_dir)
        logger.info("%s PDF-Dateien gefunden.", len(pdf_paths))
        for batch_start in range(0, len(pdf_paths), self.config["batch_size"]):
            batch = pdf_paths[batch_start : batch_start + self.config["batch_size"]]
            for pdf_path in batch:
                pdf_hash = file_hash(pdf_path)
                if pdf_hash in processed_hashes:
                    report.skipped += 1
                    logger.info("Überspringe bereits verarbeitete Datei: %s", pdf_path.name)
                    continue
                attempts = 0
                last_error: Exception | None = None
                while attempts <= self.config["max_retries"]:
                    attempts += 1
                    result = self.process_single(str(pdf_path))
                    if result.success:
                        report.processed += 1
                        report.results.append(result)
                        self.save_hash(str(pdf_path))
                        processed_hashes.add(pdf_hash)
                        break
                    last_error = RuntimeError(result.error_message or "Unbekannter Fehler")
                    logger.warning(
                        "Fehler beim Verarbeiten von %s (Versuch %s/%s): %s",
                        pdf_path.name,
                        attempts,
                        self.config["max_retries"] + 1,
                        result.error_message,
                    )
                else:
                    pass
                if last_error and (not report.results or report.results[-1].pdf_path != str(pdf_path)):
                    report.failed += 1
                    self.error_queue.add(str(pdf_path), last_error, "pipeline", attempts)
                    report.results.append(
                        ProcessingResult(
                            pdf_path=str(pdf_path),
                            success=False,
                            doc_type="sonstig",
                            chunks_created=0,
                            tables_found=0,
                            error_message=str(last_error),
                            processing_time_s=0.0,
                        )
                    )
        logger.info(
            "Batch abgeschlossen: verarbeitet=%s, übersprungen=%s, fehlgeschlagen=%s",
            report.processed,
            report.skipped,
            report.failed,
        )
        return report

    def process_single(self, pdf_path: str) -> ProcessingResult:
        start = time.perf_counter()
        logger.info("Verarbeite: %s", Path(pdf_path).name)
        try:
            classification = self.classifier.classify(pdf_path)
            extraction = self.extractor.extract(pdf_path)
            extracted_text = extraction.fliesstext
            if extraction.vertikale_header:
                extracted_text = (
                    f"Vertikale Header: {', '.join(extraction.vertikale_header)}\n\n{extracted_text}"
                ).strip()
            normalized = self.normalizer.normalize_document(extracted_text, extraction.tabellen)
            chunks = []
            text_meta = ChunkMeta(
                quelldatei=Path(pdf_path).name,
                seite=1,
                doc_type=classification.doc_type,
                tabellen_typ=None,
                extraktions_methode=extraction.extraktions_methode,
            )
            chunks.extend(self.chunker.chunk_fliesstext(normalized.fliesstext, text_meta))
            for dataframe in normalized.tabellen:
                table_type, _ = self.table_detector.detect_type(dataframe, classification.doc_type)
                table_meta = ChunkMeta(
                    quelldatei=Path(pdf_path).name,
                    seite=1,
                    doc_type=classification.doc_type,
                    tabellen_typ=table_type,
                    extraktions_methode=extraction.extraktions_methode,
                )
                chunks.extend(self.chunker.chunk_tabelle(dataframe, table_meta))
            logger.info("%s Chunks erzeugt", len(chunks))
            self.embedder.add_chunks(chunks)
            elapsed = time.perf_counter() - start
            logger.info("✓ %s — %s Chunks, %.1fs", Path(pdf_path).name, len(chunks), elapsed)
            return ProcessingResult(
                pdf_path=pdf_path,
                success=True,
                doc_type=classification.doc_type,
                chunks_created=len(chunks),
                tables_found=len(normalized.tabellen),
                error_message=None,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ProcessingResult(
                pdf_path=pdf_path,
                success=False,
                doc_type="sonstig",
                chunks_created=0,
                tables_found=0,
                error_message=str(exc),
                processing_time_s=elapsed,
            )

    def get_processed_hashes(self) -> set[str]:
        return set(load_json(self.config["processed_index_file"], default=[]))

    def save_hash(self, pdf_path: str) -> None:
        hashes = self.get_processed_hashes()
        hashes.add(file_hash(pdf_path))
        write_json(self.config["processed_index_file"], sorted(hashes))
