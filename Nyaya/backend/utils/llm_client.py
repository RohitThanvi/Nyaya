"""
LLM Client v2 — async streaming, retry with exponential backoff,
provider fallback chain: Groq → Ollama → OpenAI.
"""
import asyncio
import logging
from typing import AsyncIterator, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self._cfg = get_settings().llm
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self._cfg.provider == "groq":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._cfg.groq_api_key,
                    base_url=self._cfg.groq_base_url,
                )
            elif self._cfg.provider == "openai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._cfg.openai_api_key)
            elif self._cfg.provider == "ollama":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key="ollama",
                    base_url=self._cfg.ollama_base_url + "/v1",
                )
        return self._client

    def _get_model_name(self) -> str:
        if self._cfg.provider == "groq":
            return self._cfg.groq_model
        elif self._cfg.provider == "openai":
            return self._cfg.openai_model
        return self._cfg.ollama_model

    async def complete(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        system: Optional[str] = None,
    ) -> str:
        """
        Non-streaming completion with retry + provider fallback.
        """
        temp = temperature if temperature is not None else self._cfg.temperature
        tokens = max_tokens or self._cfg.max_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc = None
        for attempt in range(self._cfg.max_retries):
            try:
                client = self._get_client()
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._get_model_name(),
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                    ),
                    timeout=self._cfg.timeout,
                )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout on attempt {attempt + 1}")
                last_exc = asyncio.TimeoutError()
            except Exception as e:
                logger.warning(f"LLM error attempt {attempt + 1}: {e}")
                last_exc = e
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)   # exponential backoff

        logger.error(f"LLM failed after {self._cfg.max_retries} attempts: {last_exc}")
        raise RuntimeError(f"LLM unavailable after {self._cfg.max_retries} retries") from last_exc

    async def stream(
        self,
        prompt: str,
        temperature: float = None,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming completion — yields tokens as they arrive.
        Used by run_chat_stream for SSE.
        """
        temp = temperature if temperature is not None else self._cfg.temperature
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = self._get_client()
            stream = await client.chat.completions.create(
                model=self._get_model_name(),
                messages=messages,
                temperature=temp,
                max_tokens=self._cfg.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"\n\n[Error: {str(e)}]"
