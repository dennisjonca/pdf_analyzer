from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalisierterChunk:
    text: str
    quelldatei: str
    doc_type: str
    gesamtbetrag: Optional[float] = None
    rechnungsnummer: Optional[str] = None
    datum: Optional[str] = None
    faelligkeitsdatum: Optional[str] = None
    mwst_betrag: Optional[float] = None
    lieferant: Optional[str] = None
    kunde: Optional[str] = None
    seite: int = 1
    tabellen_typ: Optional[str] = None
    chunk_typ: str = "fliesstext"
    extraktions_methode: str = "digital"
    original_spalten: list[str] = field(default_factory=list)
