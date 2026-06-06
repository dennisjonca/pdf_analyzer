from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG
from utils.logger import get_logger


logger = get_logger("retriever")

PROMPT_TEMPLATE = """
Du bist ein präziser Datenextraktions-Assistent.
Beantworte die Frage AUSSCHLIESSLICH anhand der folgenden Kontextabschnitte.
Wenn die Antwort nicht im Kontext steht, antworte: "Nicht im Kontext gefunden."

Kontext:
{context}

Frage: {question}

Antworte auf Deutsch. Wenn Zahlen gefragt sind, gib sie strukturiert als JSON aus.
"""


@dataclass
class RetrievalResponse:
    answer: str
    sources: list[dict]


class Retriever:
    def __init__(self, embedder, llm_model: str | None = None, config: dict | None = None) -> None:
        self.embedder = embedder
        self.config = config or CONFIG
        self.llm_model = llm_model or self.config["llm_model"]

    def suche(self, frage: str, filter: dict | None = None, top_k: int | None = None) -> list[tuple]:
        requested_top_k = top_k or self.config["top_k"]
        results = self.embedder.similarity_search(frage, requested_top_k, metadata_filter=filter)
        threshold = self.config["score_threshold"]
        filtered = []
        for document, score in results:
            if score is None or score >= threshold:
                filtered.append((document, score))
        return filtered

    def beantworte(self, frage: str, filter: dict | None = None, top_k: int | None = None) -> RetrievalResponse:
        documents = self.suche(frage, filter=filter, top_k=top_k)
        context = "\n\n".join(doc.page_content for doc, _ in documents)
        answer = self._generate_answer(frage, context)
        sources = []
        for document, score in documents:
            metadata = dict(document.metadata)
            metadata["score"] = score
            sources.append(metadata)
        return RetrievalResponse(answer=answer, sources=sources)

    def _generate_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "Nicht im Kontext gefunden."
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            from ollama import Client

            response = Client().generate(model=self.llm_model, prompt=prompt)
            answer = (response.get("response") or "").strip()
            return answer or "Nicht im Kontext gefunden."
        except Exception as exc:
            logger.warning("Antwortgenerierung per Ollama fehlgeschlagen: %s", exc)
            return context[:1000]
