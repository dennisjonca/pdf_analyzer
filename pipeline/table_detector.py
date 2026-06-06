from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from config import CONFIG
from utils.file_utils import load_json, write_json
from utils.logger import get_logger


logger = get_logger("table_detector")


@dataclass
class TableFingerprint:
    spaltenanzahl: int
    kopfzeile_keywords: list[str]
    hat_zahlen_spalte: bool
    hat_datum_spalte: bool
    layout_hash: str


class TableTypeDetector:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or CONFIG
        self.known_layouts = load_json(self.config["table_layout_file"], default={})

    def fingerprint(self, dataframe) -> TableFingerprint:
        columns = [str(column).strip().lower() for column in dataframe.columns]
        numeric_flags = [self._is_mostly_numeric(dataframe[column]) for column in dataframe.columns]
        date_flags = [self._contains_date_values(dataframe[column]) for column in dataframe.columns]
        layout_hash = hashlib.md5("|".join(columns).encode("utf-8")).hexdigest()[:8]
        return TableFingerprint(
            spaltenanzahl=len(columns),
            kopfzeile_keywords=columns,
            hat_zahlen_spalte=any(numeric_flags),
            hat_datum_spalte=any(date_flags),
            layout_hash=layout_hash,
        )

    def detect_type(self, dataframe, doc_type: str) -> tuple[str, TableFingerprint]:
        fingerprint = self.fingerprint(dataframe)
        table_type = self.known_layouts.get(doc_type, {}).get(fingerprint.layout_hash)
        if table_type:
            return table_type, fingerprint

        doc_layouts = self.known_layouts.setdefault(doc_type, {})
        table_type = f"unbekannt_{len(doc_layouts) + 1}"
        doc_layouts[fingerprint.layout_hash] = table_type
        write_json(self.config["table_layout_file"], self.known_layouts)
        with open(self.config["new_table_log_file"], "a", encoding="utf-8") as handle:
            handle.write(f"{doc_type}: {table_type} -> {asdict(fingerprint)}\n")
        logger.info("Neuer Tabellentyp erkannt: %s (%s)", table_type, fingerprint.layout_hash)
        return table_type, fingerprint

    @staticmethod
    def _is_mostly_numeric(series) -> bool:
        cleaned = [str(value).strip() for value in series if str(value).strip()]
        if not cleaned:
            return False
        numeric = 0
        for value in cleaned:
            candidate = value.replace(".", "").replace(",", ".").replace("€", "").replace(" ", "")
            try:
                float(candidate)
                numeric += 1
            except ValueError:
                continue
        return numeric / len(cleaned) >= 0.5

    @staticmethod
    def _contains_date_values(series) -> bool:
        from datetime import datetime

        formats = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y")
        for value in series:
            candidate = str(value).strip()
            for fmt in formats:
                try:
                    datetime.strptime(candidate, fmt)
                    return True
                except ValueError:
                    continue
        return False
