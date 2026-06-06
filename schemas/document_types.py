from __future__ import annotations


DOCUMENT_TYPES = {
    "rechnung": {
        "keywords": [
            "rechnung",
            "invoice",
            "rechnungsnr",
            "re-nr",
            "rechnungsnummer",
            "zahlbar",
            "mwst",
            "ust",
        ],
        "table_types": ["positionen", "zusammenfassung"],
    },
    "lieferschein": {
        "keywords": [
            "lieferschein",
            "delivery note",
            "lieferung",
            "lieferdatum",
            "versand",
            "tracking",
        ],
        "table_types": ["lieferpositionen"],
    },
    "angebot": {
        "keywords": [
            "angebot",
            "quotation",
            "angebotsnr",
            "gültig bis",
            "angebotsdatum",
        ],
        "table_types": ["angebotspositionen"],
    },
    "vertrag": {
        "keywords": [
            "vertrag",
            "agreement",
            "vereinbarung",
            "vertragspartner",
            "laufzeit",
            "kündigung",
        ],
        "table_types": ["konditionen"],
    },
    "bericht": {
        "keywords": ["bericht", "report", "auswertung", "zusammenfassung"],
        "table_types": ["messwerte", "kennzahlen"],
    },
    "bestellung": {
        "keywords": ["bestellung", "purchase order", "po", "bestellnummer"],
        "table_types": ["bestellpositionen"],
    },
    "sonstig": {
        "keywords": [],
        "table_types": ["unbekannt"],
    },
}


ALLOWED_DOCUMENT_TYPES = list(DOCUMENT_TYPES.keys())
