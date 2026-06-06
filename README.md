# pdf_analyzer

Lokale RAG-Pipeline für die Massenverarbeitung von PDFs mit Dokumentklassifikation, Tabellen-/Textextraktion, semantischer Normalisierung und CLI-Suche.

## Architektur

Die Implementierung folgt einer modularen Pipeline:

- `main.py` – CLI für `ingest` und `search`
- `config.py` – zentrale Parameter
- `pipeline/` – Batch-Orchestrierung, Klassifikation, Extraktion, Tabellenerkennung, Normalisierung, Chunking, Embeddings, Retrieval
- `schemas/` – Dokumenttypen, Synonym-Mapping, Zielschema
- `utils/` – Logging, Datei-Utilities, Fehler-Queue
- `data/` – Input, Chroma-Persistenz, Logs und Fehlerberichte

## Voraussetzungen

Die Pipeline ist lokal/offline ausgelegt und benötigt folgende Dienste bzw. Systempakete:

1. Python 3.10+
2. Ollama lokal:
   - `ollama serve`
   - `ollama pull llama3.2`
   - `ollama pull nomic-embed-text`
3. Ghostscript für Camelot:
   - Ubuntu/Debian: `sudo apt install ghostscript`
   - macOS: `brew install ghostscript`
4. Python-Abhängigkeiten:

```bash
pip install -r requirements.txt
```

Hinweis: `easyocr` lädt beim ersten Lauf Modelle lokal herunter. Das ist einmalig.

## Nutzung

### PDFs ingestieren

```bash
python main.py ingest --input data/input/
python main.py ingest --input data/input/ --reset
```

Verhalten:

- rekursives PDF-Scanning
- Hash-basierte Skip-Logik für bereits verarbeitete PDFs
- Batch-Verarbeitung mit Retry-Logik
- fehlgeschlagene PDFs werden nach `data/errors/` kopiert
- Fehlerbericht landet in `data/errors/fehlerbericht.json`

### Suchen

```bash
python main.py search --query "Was ist der Gesamtbetrag der Rechnung vom März 2024?"
python main.py search --query "Alle Positionen von Lieferant Müller GmbH" --type rechnung
python main.py search --query "Zahlungsziel" --type rechnung --top-k 10
```

## Hinweise zur Extraktion

- Digitale PDFs: Text via PyMuPDF, Tabellen via Camelot (`lattice`, dann `stream`), danach Docling-Fallback
- Gescannte PDFs: OCR via EasyOCR auf gerenderten Seiten
- Vertikale Textzeilen werden separat gesammelt
- OCR-basierte Tabellenerkennung ist absichtlich pragmatisch gehalten: die Pipeline extrahiert derzeit robust OCR-Text, aber keine vollwertige Tabellenrekonstruktion aus OCR-Layouts. Diese Einschränkung ist dokumentiert, damit die Pipeline lokal stabil bleibt.

## Logging und Datenablage

- Batch-Logs: `data/logs/batch_*.log`
- erkannte neue Tabellenlayouts: `data/logs/neue_tabellentypen.log`
- Chroma-Persistenz: `data/chroma_db/`

## Bekannte Grenzen

- Für beste Ergebnisse sollten Camelot, Docling, EasyOCR und Ollama lokal korrekt installiert sein.
- Wenn Ollama nicht läuft, fallen Klassifikation/Antwortgenerierung robust zurück, aber die Ergebnisse werden einfacher.
- Tabellen-Typen werden zunächst als `unbekannt_N` gespeichert, bis ein Layout wiederverwendet oder manuell nachgepflegt wird.
