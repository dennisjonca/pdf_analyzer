from __future__ import annotations

from copy import deepcopy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "input_dir": str(BASE_DIR / "data" / "input"),
    "chroma_dir": str(BASE_DIR / "data" / "chroma_db"),
    "log_dir": str(BASE_DIR / "data" / "logs"),
    "error_dir": str(BASE_DIR / "data" / "errors"),
    "processed_index_file": str(BASE_DIR / "data" / "processed_hashes.json"),
    "table_layout_file": str(BASE_DIR / "data" / "logs" / "table_layouts.json"),
    "new_table_log_file": str(BASE_DIR / "data" / "logs" / "neue_tabellentypen.log"),
    "llm_model": "llama3.2",
    "embed_model": "nomic-embed-text",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "ocr_languages": ["de", "en"],
    "ocr_dpi": 300,
    "camelot_flavor": "lattice",
    "min_table_rows": 2,
    "top_k": 5,
    "score_threshold": 0.3,
    "batch_size": 10,
    "max_retries": 2,
    "embedding_batch_size": 100,
    "classification_char_limit": 500,
    "ocr_table_extraction_enabled": False,
    "collection_name": "rag_pipeline",
}


def get_config() -> dict:
    return deepcopy(CONFIG)
