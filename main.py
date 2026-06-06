from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import CONFIG
from pipeline.batch_processor import BatchProcessor, BatchReport
from pipeline.embedder import ChromaEmbedder
from pipeline.retriever import RetrievalResponse, Retriever
from utils.file_utils import ensure_directory, require_directory, reset_directory_contents
from utils.logger import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lokale PDF-RAG-Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="PDFs klassifizieren, extrahieren und indexieren")
    ingest_parser.add_argument("--input", default=CONFIG["input_dir"], help="Pfad zum Eingabeordner")
    ingest_parser.add_argument("--reset", action="store_true", help="Leert den Chroma-Index vor dem Import")

    search_parser = subparsers.add_parser("search", help="Semantische Suche im lokalen Index")
    search_parser.add_argument("--query", required=True, help="Freitextsuchanfrage")
    search_parser.add_argument("--type", dest="doc_type", help="Optionaler Dokumenttypfilter")
    search_parser.add_argument("--top-k", type=int, default=CONFIG["top_k"], help="Anzahl Treffer")
    return parser


def ingest_documents(input_dir: str, reset: bool, config: dict | None = None) -> BatchReport:
    current_config = config or CONFIG
    validated_input_dir = require_directory(input_dir)
    setup_logging(current_config["log_dir"])
    ensure_directory(current_config["error_dir"])
    ensure_directory(current_config["chroma_dir"])
    processor = BatchProcessor(current_config)
    if reset:
        processor.embedder.reset()
        processed_index = Path(current_config["processed_index_file"])
        if processed_index.exists():
            processed_index.unlink()
        reset_directory_contents(current_config["error_dir"])
    return processor.run(str(validated_input_dir))


def run_ingest(input_dir: str, reset: bool) -> int:
    try:
        report = ingest_documents(input_dir, reset)
    except RuntimeError as exc:
        print(f"Fehler beim Ingest: {exc}")
        return 1
    print(
        json.dumps(
            {
                "processed": report.processed,
                "skipped": report.skipped,
                "failed": report.failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.failed == 0 else 1


def search_documents(
    query: str,
    doc_type: str | None,
    top_k: int,
    config: dict | None = None,
) -> RetrievalResponse:
    current_config = config or CONFIG
    embedder = ChromaEmbedder(current_config)
    retriever = Retriever(embedder, config=current_config)
    filters = {"doc_type": doc_type} if doc_type else None
    return retriever.beantworte(query, filter=filters, top_k=top_k)


def run_search(query: str, doc_type: str | None, top_k: int) -> int:
    try:
        response = search_documents(query, doc_type, top_k)
    except RuntimeError as exc:
        print(f"Fehler bei der Suche: {exc}")
        return 1
    print(f"Frage: {query}\n")
    print("Antwort:")
    print(response.answer)
    print("\nQuellen:")
    if not response.sources:
        print("  Keine Treffer.")
        return 0
    for index, source in enumerate(response.sources, start=1):
        score = source.get("score")
        score_text = f"{score:.2f}" if isinstance(score, float) else "n/a"
        print(
            f"  [{index}] {source.get('quelldatei', 'unbekannt')} — "
            f"Seite {source.get('seite', '?')} (Ähnlichkeit: {score_text})"
        )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "ingest":
        return run_ingest(args.input, args.reset)
    if args.command == "search":
        return run_search(args.query, args.doc_type, args.top_k)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
