from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import CONFIG
from pipeline.batch_processor import BatchProcessor
from pipeline.embedder import ChromaEmbedder
from pipeline.retriever import Retriever
from utils.file_utils import ensure_directory, reset_directory_contents
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


def run_ingest(input_dir: str, reset: bool) -> int:
    setup_logging(CONFIG["log_dir"])
    ensure_directory(CONFIG["input_dir"])
    ensure_directory(CONFIG["error_dir"])
    ensure_directory(CONFIG["chroma_dir"])
    processor = BatchProcessor()
    if reset:
        processor.embedder.reset()
        processed_index = Path(CONFIG["processed_index_file"])
        if processed_index.exists():
            processed_index.unlink()
        reset_directory_contents(CONFIG["error_dir"])
    try:
        report = processor.run(input_dir)
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


def run_search(query: str, doc_type: str | None, top_k: int) -> int:
    embedder = ChromaEmbedder()
    retriever = Retriever(embedder)
    filters = {"doc_type": doc_type} if doc_type else None
    try:
        response = retriever.beantworte(query, filter=filters, top_k=top_k)
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
