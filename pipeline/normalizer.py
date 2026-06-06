from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from config import CONFIG
from schemas.field_mapping import FIELD_MAPPING, REVERSE_FIELD_MAPPING
from utils.logger import get_logger


logger = get_logger("normalizer")


@dataclass
class NormalizedDocument:
    fliesstext: str
    tabellen: list
    column_mappings: list[dict[str, str]]


class SemanticNormalizer:
    def __init__(self, llm_model: str | None = None) -> None:
        self.llm_model = llm_model or CONFIG["llm_model"]
        self._llm_cache: dict[str, str] = {}

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def normalize_table(self, dataframe):
        original_columns = [str(column) for column in dataframe.columns]
        mapped_columns = {column: self._map_column(str(column)) for column in dataframe.columns}
        normalized = dataframe.rename(columns=mapped_columns).copy()
        for column in normalized.columns:
            if column in {"gesamtbetrag", "mwst_betrag", "menge", "einzelpreis", "gesamtpreis"}:
                normalized[column] = normalized[column].apply(self._to_float)
            elif column in {"datum", "faelligkeitsdatum"}:
                normalized[column] = normalized[column].apply(self._to_iso_date)
            else:
                normalized[column] = normalized[column].apply(self._clean_string)
        return normalized, mapped_columns, original_columns

    def normalize_document(self, text: str, tables: list) -> NormalizedDocument:
        normalized_tables = []
        mappings = []
        for dataframe in tables:
            normalized, mapped_columns, _ = self.normalize_table(dataframe)
            normalized_tables.append(normalized)
            mappings.append(mapped_columns)
        return NormalizedDocument(
            fliesstext=self.normalize_text(text),
            tabellen=normalized_tables,
            column_mappings=mappings,
        )

    def _map_column(self, column_name: str) -> str:
        normalized_name = self._clean_string(column_name).lower()
        if normalized_name in REVERSE_FIELD_MAPPING:
            return REVERSE_FIELD_MAPPING[normalized_name]
        if normalized_name in self._llm_cache:
            return self._llm_cache[normalized_name]
        llm_mapping = self._map_with_llm([column_name]).get(column_name, column_name)
        self._llm_cache[normalized_name] = llm_mapping
        return llm_mapping

    def _map_with_llm(self, columns: list[str]) -> dict[str, str]:
        prompt = (
            "Mappe diese Spalten auf das Zielschema.\n"
            'Antworte als JSON: {"quellspalte": "zielfeld_oder_unbekannt"}\n\n'
            f"Quellspalten: {columns}\n"
            f"Erlaubte Zielfelder: {list(FIELD_MAPPING.keys())}"
        )
        try:
            from ollama import Client

            response = Client().generate(model=self.llm_model, prompt=prompt)
            payload = json.loads((response.get("response") or "{}").strip())
            return {
                column: payload.get(column, column)
                if payload.get(column, column) in FIELD_MAPPING or payload.get(column) == "unbekannt"
                else column
                for column in columns
            }
        except Exception as exc:
            logger.warning("LLM-Fallback für Spalten fehlgeschlagen: %s", exc)
            return {column: column for column in columns}

    @staticmethod
    def _clean_string(value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"[^\d,\.\-]", "", text)
        if text.count(",") == 1 and text.count(".") >= 1:
            text = text.replace(".", "").replace(",", ".")
        elif text.count(",") == 1 and text.count(".") == 0:
            text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _to_iso_date(value):
        text = str(value or "").strip()
        if not text:
            return None
        formats = ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
        return text
