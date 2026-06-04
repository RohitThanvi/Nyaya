"""
LLM abstraction layer using LiteLLM.
Single interface for Groq, Ollama, and OpenAI.
Supports streaming, structured outputs, and automatic retry.
"""
import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm import acompletion
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# Suppress litellm verbose logging
litellm.suppress_debug_info = True
litellm.set_verbose = False


class LLMClient:
    """
    Unified LLM client.
    Swap providers by changing LLM_PROVIDER env var.
    Supports: groq | ollama | openai
    """

    def __init__(self):
        settings = get_settings().llm
        self._settings = settings
        self._model = self._resolve_model()
        self._setup_provider()

    def _resolve_model(self) -> str:
        s = self._settings
        if s.provider == "groq":
            return f"groq/{s.groq_model}"
        elif s.provider == "ollama":
            return f"ollama/{s.ollama_model}"
        elif s.provider == "openai":
            return f"openai/{s.openai_model}"
        else:
            raise ValueError(f"Unknown LLM provider: {s.provider}")

    def _setup_provider(self) -> None:
        s = self._settings
        if s.provider == "groq" and s.groq_api_key:
            litellm.groq_key = s.groq_api_key
        elif s.provider == "openai" and s.openai_api_key:
            litellm.openai_key = s.openai_api_key
        elif s.provider == "ollama":
            litellm.ollama_base_url = s.ollama_base_url

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Single completion with retry logic."""
        t = temperature if temperature is not None else self._settings.temperature
        mt = max_tokens or self._settings.max_tokens

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": t,
            "max_tokens": mt,
            "timeout": self._settings.timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True,
        ):
            with attempt:
                response = await acompletion(**kwargs)
                return response.choices[0].message.content

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion for real-time UI updates."""
        t = temperature if temperature is not None else self._settings.temperature
        mt = max_tokens or self._settings.max_tokens

        response = await acompletion(
            model=self._model,
            messages=messages,
            temperature=t,
            max_tokens=mt,
            timeout=self._settings.timeout,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def complete_with_json(
        self,
        messages: List[Dict[str, str]],
        expected_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Completion that parses JSON output.
        Falls back to extracting JSON from markdown code blocks.
        """
        raw = await self.complete(messages, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting from ```json ... ``` blocks
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")


# Module-level singleton
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
