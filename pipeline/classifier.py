from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG
from schemas.document_types import ALLOWED_DOCUMENT_TYPES, DOCUMENT_TYPES
from utils.logger import get_logger


logger = get_logger("classifier")


@dataclass
class ClassificationResult:
    doc_type: str
    confidence: int
    method: str


class DocumentClassifier:
    def __init__(self, llm_model: str | None = None, char_limit: int | None = None) -> None:
        self.llm_model = llm_model or CONFIG["llm_model"]
        self.char_limit = char_limit or CONFIG["classification_char_limit"]

    def classify(self, pdf_path: str) -> ClassificationResult:
        sample = self._read_first_text(pdf_path)
        rule_match = self._classify_rule_based(sample)
        if rule_match.confidence >= 2:
            logger.info(
                "Typ erkannt: %s (regelbasiert, Score=%s)",
                rule_match.doc_type,
                rule_match.confidence,
            )
            return rule_match
        llm_result = self._classify_with_llm(sample)
        logger.info("Typ erkannt: %s (%s)", llm_result.doc_type, llm_result.method)
        return llm_result

    def _read_first_text(self, pdf_path: str) -> str:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF nicht installiert, Klassifikation fällt auf sonstig zurück.")
            return ""
        document = fitz.open(pdf_path)
        try:
            if document.page_count == 0:
                return ""
            text = document.load_page(0).get_text() or ""
            return text[: self.char_limit]
        finally:
            document.close()

    def _classify_rule_based(self, text: str) -> ClassificationResult:
        haystack = text.lower()
        best_type = "sonstig"
        best_score = 0
        for doc_type, config in DOCUMENT_TYPES.items():
            score = sum(1 for keyword in config["keywords"] if keyword in haystack)
            if score > best_score:
                best_type = doc_type
                best_score = score
        return ClassificationResult(doc_type=best_type, confidence=best_score, method="regelbasiert")

    def _classify_with_llm(self, text: str) -> ClassificationResult:
        prompt = (
            "Klassifiziere dieses Dokument. Erlaubte Kategorien:\n"
            "rechnung | lieferschein | angebot | vertrag | bericht | bestellung | sonstig\n\n"
            "Antworte NUR mit dem Kategorienamen.\n\n"
            f"Text: {text}"
        )
        try:
            from ollama import Client

            client = Client()
            response = client.generate(model=self.llm_model, prompt=prompt)
            raw_answer = (response.get("response") or "").strip().lower()
            doc_type = raw_answer.splitlines()[0].strip()
            if doc_type not in ALLOWED_DOCUMENT_TYPES:
                doc_type = "sonstig"
            return ClassificationResult(doc_type=doc_type, confidence=0, method="llm")
        except Exception as exc:
            logger.warning("LLM-Klassifikation fehlgeschlagen (%s), verwende sonstig.", exc)
            return ClassificationResult(doc_type="sonstig", confidence=0, method="llm_fallback")
