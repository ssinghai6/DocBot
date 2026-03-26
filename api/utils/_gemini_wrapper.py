"""Minimal LangChain-compatible wrapper around google-generativeai SDK.

Avoids the broken langchain-google-genai package (pydantic_v1 import error).
Only implements the subset of BaseChatModel needed for fallback usage.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


def _messages_to_genai_format(messages: List[BaseMessage]) -> tuple[Optional[str], list[dict]]:
    """Convert LangChain messages to google-generativeai format.

    Returns (system_instruction, contents) where contents is a list of
    {"role": "user"|"model", "parts": [text]} dicts.
    """
    system_instruction = None
    contents: list[dict] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_instruction = str(msg.content)
        elif isinstance(msg, HumanMessage):
            contents.append({"role": "user", "parts": [str(msg.content)]})
        elif isinstance(msg, AIMessage):
            contents.append({"role": "model", "parts": [str(msg.content)]})

    return system_instruction, contents


class GeminiChatModel(BaseChatModel):
    """Thin wrapper around google.generativeai for LangChain compatibility."""

    model: str = "gemini-2.5-flash"
    api_key: str = ""
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return "gemini-wrapper"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via google-generativeai SDK."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)

        system_instruction, contents = _messages_to_genai_format(messages)

        gen_config = genai.GenerationConfig(
            temperature=self.temperature,
            stop_sequences=stop or [],
        )

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_instruction,
            generation_config=gen_config,
        )

        response = model.generate_content(contents)
        text = response.text or ""

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation — delegates to sync in a thread."""
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(
            None, self._generate, messages, stop,
        )
