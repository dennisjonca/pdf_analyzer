from __future__ import annotations

import streamlit as st

from config import CONFIG
from main import ingest_documents, search_documents


st.set_page_config(page_title="PDF Analyzer", layout="wide")
st.title("PDF Analyzer")
st.caption(
    f"Lokale PDF-RAG-Pipeline mit Ollama ({CONFIG['llm_model']}) und "
    f"{CONFIG['embed_model']}."
)

with st.sidebar:
    st.header("Ingest")
    input_dir = st.text_input("Input-Verzeichnis", value=CONFIG["input_dir"])
    reset_index = st.checkbox("Index vor dem Import zurücksetzen", value=False)
    ingest_button = st.button("Ingest starten", use_container_width=True)

st.subheader("Suche")
query = st.text_input("Frage", placeholder="Was ist der Gesamtbetrag der Rechnung vom März 2024?")
doc_type = st.text_input("Dokumenttyp (optional)", placeholder="rechnung")
top_k = st.slider("Anzahl Treffer", min_value=1, max_value=20, value=CONFIG["top_k"])
search_button = st.button("Suche starten", type="primary")

if ingest_button:
    with st.spinner("PDFs werden verarbeitet ..."):
        try:
            report = ingest_documents(input_dir, reset_index)
        except RuntimeError as exc:
            st.error(f"Ingest fehlgeschlagen: {exc}")
        else:
            st.success("Ingest abgeschlossen.")
            st.json(
                {
                    "processed": report.processed,
                    "skipped": report.skipped,
                    "failed": report.failed,
                }
            )

if search_button:
    if not query.strip():
        st.warning("Bitte eine Suchanfrage eingeben.")
    else:
        with st.spinner("Suche läuft ..."):
            try:
                response = search_documents(query, doc_type.strip() or None, top_k)
            except RuntimeError as exc:
                st.error(f"Suche fehlgeschlagen: {exc}")
            else:
                st.markdown("### Antwort")
                st.write(response.answer)
                st.markdown("### Quellen")
                if response.sources:
                    st.dataframe(response.sources, use_container_width=True)
                else:
                    st.info("Keine Treffer gefunden.")
