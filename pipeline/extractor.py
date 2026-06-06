from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from config import CONFIG
from utils.logger import get_logger


logger = get_logger("extractor")


@dataclass
class ExtractionResult:
    fliesstext: str
    tabellen: list[Any]
    vertikale_header: list[str]
    seiten_anzahl: int
    extraktions_methode: str


class PDFExtractor:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or CONFIG

    def extract(self, pdf_path: str) -> ExtractionResult:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF ist erforderlich für die PDF-Extraktion.") from exc

        document = fitz.open(pdf_path)
        try:
            page_count = document.page_count
            digital_text, vertical_headers = self._extract_text_and_vertical_headers(document)
            if digital_text.strip():
                tables, table_method = self._extract_digital_tables(pdf_path)
                method = table_method if table_method == "docling" else "digital"
                logger.info("%s Tabellen gefunden, Methode: %s", len(tables), method)
                return ExtractionResult(
                    fliesstext=digital_text,
                    tabellen=tables,
                    vertikale_header=vertical_headers,
                    seiten_anzahl=page_count,
                    extraktions_methode=method,
                )

            ocr_text = self._extract_ocr_text(document)
            logger.info("OCR-Extraktion ausgeführt, Tabellen aus OCR werden nicht zuverlässig erkannt.")
            return ExtractionResult(
                fliesstext=ocr_text,
                tabellen=[],
                vertikale_header=[],
                seiten_anzahl=page_count,
                extraktions_methode="ocr",
            )
        finally:
            document.close()

    def _extract_text_and_vertical_headers(self, document) -> tuple[str, list[str]]:
        text_parts: list[str] = []
        vertical_headers: list[str] = []
        for page in document:
            raw_dict = page.get_text("rawdict")
            page_text = page.get_text() or ""
            if page_text:
                text_parts.append(page_text)
            for block in raw_dict.get("blocks", []):
                for line in block.get("lines", []):
                    direction = tuple(line.get("dir", (1, 0)))
                    if direction in {(0, 1), (0, -1)}:
                        spans = []
                        for span in line.get("spans", []):
                            span_text = "".join(char.get("c", "") for char in span.get("chars", []))
                            if span_text.strip():
                                spans.append(span_text.strip())
                        if spans:
                            vertical_headers.append(" ".join(spans))
        deduped_headers = list(dict.fromkeys(vertical_headers))
        return "\n".join(text_parts).strip(), deduped_headers

    def _extract_digital_tables(self, pdf_path: str) -> tuple[list[Any], str]:
        tables = self._try_camelot(pdf_path, "lattice")
        if tables:
            return tables, "digital"
        tables = self._try_camelot(pdf_path, "stream")
        if tables:
            return tables, "digital"
        return self._try_docling(pdf_path), "docling"

    def _try_camelot(self, pdf_path: str, flavor: str) -> list[Any]:
        try:
            import camelot
        except ImportError:
            return []

        kwargs = {"flavor": flavor, "pages": "all"}
        if flavor == "stream":
            kwargs.update({"edge_tol": 50, "row_tol": 10})
        try:
            tables = camelot.read_pdf(pdf_path, **kwargs)
            return [
                table.df
                for table in tables
                if getattr(table, "df", None) is not None
                and len(table.df.index) >= self.config["min_table_rows"]
            ]
        except Exception as exc:
            logger.warning("Camelot %s fehlgeschlagen: %s", flavor, exc)
            return []

    def _try_docling(self, pdf_path: str) -> list[Any]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            return []

        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            tables = getattr(result.document, "tables", []) or []
            extracted_tables = []
            for table in tables:
                if hasattr(table, "export_to_dataframe"):
                    dataframe = table.export_to_dataframe()
                else:
                    dataframe = getattr(table, "data", None)
                if dataframe is not None and len(dataframe.index) >= self.config["min_table_rows"]:
                    extracted_tables.append(dataframe)
            return extracted_tables
        except Exception as exc:
            logger.warning("Docling-Fallback fehlgeschlagen: %s", exc)
            return []

    def _extract_ocr_text(self, document) -> str:
        try:
            import fitz
            import easyocr
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("EasyOCR und Pillow sind für gescannte PDFs erforderlich.") from exc

        reader = easyocr.Reader(self.config["ocr_languages"], gpu=False)
        zoom = self.config["ocr_dpi"] / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        page_texts: list[str] = []
        for page in document:
            pix = page.get_pixmap(matrix=matrix)
            image = Image.open(BytesIO(pix.tobytes("png")))
            ocr_result = reader.readtext(image)
            page_texts.append("\n".join(item[1] for item in ocr_result))
        return "\n".join(page_texts).strip()
