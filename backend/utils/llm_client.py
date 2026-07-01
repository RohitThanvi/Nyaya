"""
LLM Client v2 — unified interface used by all agents.

complete()           — non-streaming, returns str
complete_with_json() — non-streaming, returns parsed dict (used by QUA + LegalMapping)
complete_messages()  — non-streaming, takes messages list directly
stream()             — async generator, yields str tokens

get_llm_client()     — module-level singleton factory (used by old agents)
"""
import asyncio
import json
import logging
import re
import threading
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_singleton: Optional["LLMClient"] = None
_singleton_lock = threading.Lock()


def get_llm_client() -> "LLMClient":
    """Module-level singleton — thread-safe for concurrent Celery workers."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = LLMClient()
    return _singleton


class LLMClient:
    def __init__(self):
        self._cfg = get_settings().llm
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            if self._cfg.provider == "groq":
                self._client = AsyncOpenAI(
                    api_key=self._cfg.groq_api_key,
                    base_url=self._cfg.groq_base_url,
                )
            elif self._cfg.provider == "openai":
                self._client = AsyncOpenAI(api_key=self._cfg.openai_api_key)
            else:  # ollama
                self._client = AsyncOpenAI(
                    api_key="ollama",
                    base_url=self._cfg.ollama_base_url + "/v1",
                )
        return self._client

    def _model(self) -> str:
        if self._cfg.provider == "groq":
            return self._cfg.groq_model
        if self._cfg.provider == "openai":
            return self._cfg.openai_model
        return self._cfg.ollama_model

    # ── Core: messages list ────────────────────────────────────────────────

    async def complete_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a messages list directly. Foundation for all other methods."""
        temp = temperature if temperature is not None else self._cfg.temperature
        tokens = max_tokens or self._cfg.max_tokens
        last_exc = None

        for attempt in range(self._cfg.max_retries):
            try:
                client = self._get_client()
                kwargs: Dict[str, Any] = dict(
                    model=self._model(),
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                # Groq supports JSON mode — use it for reliability
                if response_format:
                    kwargs["response_format"] = response_format

                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self._cfg.timeout,
                )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout attempt {attempt + 1}")
                last_exc = asyncio.TimeoutError()
            except Exception as e:
                logger.warning(f"LLM error attempt {attempt + 1}: {e}")
                last_exc = e
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"LLM unavailable after {self._cfg.max_retries} retries: {last_exc}")

    # ── complete() — simple prompt → str ──────────────────────────────────

    async def complete(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        system: Optional[str] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.complete_messages(messages, temperature=temperature, max_tokens=max_tokens)

    # ── complete_with_json() — messages list → parsed dict ─────────────────
    # Called by QueryUnderstandingAgent and LegalMappingAgent

    async def complete_with_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> Dict:
        """
        Send messages and parse response as JSON.
        Uses JSON mode on Groq/OpenAI when available.
        Falls back to extracting JSON from markdown fences.
        """
        # Try with JSON response_format first (Groq supports this)
        try:
            raw = await self.complete_messages(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback: plain completion with JSON instruction appended
            msgs = list(messages)
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1] = {
                    "role": "user",
                    "content": msgs[-1]["content"] + "\n\nRespond ONLY with valid JSON, no other text.",
                }
            raw = await self.complete_messages(
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> Dict:
        """Extract and parse JSON from LLM response, handling markdown fences."""
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        # Try direct parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Find first { ... } block
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from LLM response: {raw[:200]}")
        return {}

    # ── stream() — async generator for SSE ────────────────────────────────

    async def stream(
        self,
        prompt: str,
        temperature: float = None,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        temp = temperature if temperature is not None else self._cfg.temperature
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc = None
        for attempt in range(self._cfg.max_retries):
            try:
                client = self._get_client()
                stream = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._model(),
                        messages=messages,
                        temperature=temp,
                        max_tokens=self._cfg.max_tokens,
                        stream=True,
                    ),
                    timeout=self._cfg.timeout,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  # clean exit
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError("LLM stream timed out")
                logger.warning(f"LLM stream timeout attempt {attempt + 1}")
            except Exception as e:
                last_exc = e
                logger.warning(f"LLM stream error attempt {attempt + 1}: {e}")

            if attempt < self._cfg.max_retries - 1:
                backoff = min(2 ** attempt, 30)
                await asyncio.sleep(backoff)

        logger.error(f"LLM stream failed after {self._cfg.max_retries} retries: {last_exc}")
        yield f"\n\n[Response unavailable — please retry. Error: {last_exc}]"
