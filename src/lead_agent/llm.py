"""LiteLLM adapter. Single point for all LLM calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import instructor
import litellm
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)


class LLMSettings(BaseSettings):
    llm_provider: Literal["ollama", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@dataclass
class CallStats:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_ms: int


@dataclass
class LLMResponse(Generic[T]):
    content: T
    stats: CallStats


def _build_messages(prompt: str, system: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_stats(
    completion: Any,
    model: str,
    duration_ms: int,
    provider: str,
) -> CallStats:
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    if provider == "groq":
        try:
            cost_usd = float(litellm.completion_cost(completion_response=completion))
        except Exception:
            cost_usd = 0.0
    else:
        cost_usd = 0.0
    return CallStats(
        model=getattr(completion, "model", None) or model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
    )


class LLMClient:
    def __init__(
        self,
        model: str,
        extra_kwargs: dict[str, str],
        provider: str,
        _acompletion: Any = None,
        _instructor_client: Any = None,
    ) -> None:
        self._model = model
        self._extra_kwargs = extra_kwargs
        self._provider = provider
        _ac = _acompletion or litellm.acompletion
        self._acompletion = _ac
        self._instructor = _instructor_client or instructor.from_litellm(_ac)

    async def ask(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
    ) -> LLMResponse[str]:
        """Raw text completion for query generation, filtering, and scoring."""
        messages = _build_messages(prompt, system)
        start = time.monotonic()
        completion = await self._acompletion(
            model=self._model,
            messages=messages,
            temperature=temperature,
            **self._extra_kwargs,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        content: str = completion.choices[0].message.content or ""
        return LLMResponse(
            content=content,
            stats=_extract_stats(completion, self._model, duration_ms, self._provider),
        )

    async def extract(
        self,
        prompt: str,
        response_model: type[T],
        system: str = "",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> LLMResponse[T]:
        """Structured extraction returning a validated Pydantic model."""
        messages = _build_messages(prompt, system)
        start = time.monotonic()
        parsed, completion = await self._instructor.chat.completions.create_with_completion(
            model=self._model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=max_retries,
            **self._extra_kwargs,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return LLMResponse(
            content=parsed,
            stats=_extract_stats(completion, self._model, duration_ms, self._provider),
        )


def get_client() -> LLMClient:
    """Read env/settings and return a configured LLMClient. Call once at pipeline startup."""
    from pydantic import ValidationError as PydanticValidationError

    try:
        settings = LLMSettings()
    except PydanticValidationError as e:
        raise ValueError(f"Invalid LLM_PROVIDER setting:\n{e}") from e
    if settings.llm_provider == "ollama":
        model = f"ollama/{settings.ollama_model}"
        extra: dict[str, str] = {"api_base": settings.ollama_base_url}
    elif settings.llm_provider == "groq":
        model = f"groq/{settings.groq_model}"
        extra = {}
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Expected 'ollama' or 'groq'."
        )
    return LLMClient(model=model, extra_kwargs=extra, provider=settings.llm_provider)
