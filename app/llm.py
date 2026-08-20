"""Генерация ответа по найденному контексту.

Если задан ANTHROPIC_API_KEY — ответ формирует Claude по найденным чанкам.
Без ключа сервис не должен падать или врать, что "сгенерировал" ответ:
включается честный fallback-режим — он возвращает наиболее релевантный
кусок документа как есть, с пометкой источника.
"""
from __future__ import annotations

import os


SYSTEM_PROMPT = (
    "Ты отвечаешь на вопрос строго по приведённому контексту из базы знаний. "
    "Если ответа в контексте нет — так и скажи, не придумывай."
)


class AnswerGenerator:
    def __init__(self):
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def mode(self) -> str:
        return "llm" if self._client else "fallback_extractive"

    def answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "В базе знаний не нашлось релевантного фрагмента."

        if not self._client:
            best = contexts[0]
            return f"[офлайн-режим, без LLM] Наиболее релевантный фрагмент: {best}"

        context_block = "\n\n".join(f"- {c}" for c in contexts)
        message = self._client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Контекст:\n{context_block}\n\nВопрос: {question}",
                }
            ],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
