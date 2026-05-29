"""LiteLLM adapter. Single point for all LLM calls.

All calls go through `LLMClient._invoke`, which enforces a per-request timeout and
retries rate-limit errors while honoring the provider's `retry-after` hint. Groq's
free tier in particular returns 429s with a "try again in Ns" message; without
honoring that pause, fast retries exhaust before the per-minute window resets.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import instructor
import litellm
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

# Fallback pause when a rate-limit error carries no parseable retry-after.
_DEFAULT_RETRY_AFTER_SECONDS = 10.0


class LLMSettings(BaseSettings):
    llm_provider: Literal["ollama", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_request_timeout_seconds: float = 60.0
    llm_rate_limit_max_attempts: int = 3
    llm_rate_limit_max_sleep_seconds: float = 60.0

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


# ---------------------------------------------------------------------------
# Rate-limit detection and retry-after parsing
# ---------------------------------------------------------------------------

def _rate_limit_error_types() -> tuple[type, ...]:
    """Known rate-limit exception classes, gathered defensively across versions."""
    types: list[type] = []
    rate_limit = getattr(litellm, "RateLimitError", None)
    if isinstance(rate_limit, type):
        types.append(rate_limit)
    try:
        import openai

        if isinstance(openai.RateLimitError, type):
            types.append(openai.RateLimitError)
    except Exception:  # openai always present via litellm, but never hard-fail here
        pass
    return tuple(dict.fromkeys(types))


_RATE_LIMIT_ERROR_TYPES = _rate_limit_error_types()

_RETRY_AFTER_MSG_RE = re.compile(r"try again in\s+([0-9hms.\s]+)", re.IGNORECASE)
_DURATION_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def _exc_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield exc and its __cause__/__context__ ancestors, guarding against cycles."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if exc (or a wrapped cause) is a rate-limit/429 error.

    Tolerant of wrappers (e.g. instructor) that re-raise the provider error: walks the
    cause chain and falls back to a string heuristic on the rendered message.
    """
    for err in _exc_chain(exc):
        if _RATE_LIMIT_ERROR_TYPES and isinstance(err, _RATE_LIMIT_ERROR_TYPES):
            return True
        if "ratelimit" in type(err).__name__.lower():
            return True
    text = str(exc).lower()
    return "rate_limit" in text or "ratelimiterror" in text or "too many requests" in text


def _parse_retry_after_value(raw: object) -> float | None:
    """Parse a numeric `Retry-After` header value (seconds). HTTP-date form is ignored."""
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _parse_retry_after_message(text: str) -> float | None:
    """Extract seconds from a 'try again in 8.855s' / '1m30s' style message."""
    match = _RETRY_AFTER_MSG_RE.search(text)
    if not match:
        return None
    total = 0.0
    found = False
    for value, unit in _DURATION_TOKEN_RE.findall(match.group(1)):
        found = True
        total += float(value) * _UNIT_SECONDS[unit.lower()]
    return total if found else None


def _retry_after_seconds(exc: BaseException, default: float) -> float:
    """Seconds to wait before retrying: prefer Retry-After headers, then the message."""
    for err in _exc_chain(exc):
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers is None:
            continue
        try:
            raw = headers.get("retry-after")
        except AttributeError:
            raw = None
        value = _parse_retry_after_value(raw)
        if value is not None:
            return value
    value = _parse_retry_after_message(str(exc))
    return value if value is not None else default


def _build_messages(prompt: str, system: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_stats(
    completion: object,
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
        _acompletion: Callable[..., Awaitable[Any]] | None = None,
        _instructor_client: object = None,
        request_timeout: float | None = 60.0,
        max_attempts: int = 3,
        max_sleep: float = 60.0,
        _sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._model = model
        self._extra_kwargs = extra_kwargs
        self._provider = provider
        _ac = _acompletion or litellm.acompletion
        self._acompletion = _ac
        self._instructor = _instructor_client or instructor.from_litellm(_ac)
        self._request_timeout = request_timeout
        self._max_attempts = max(1, max_attempts)
        self._max_sleep = max_sleep
        self._sleep = _sleep or asyncio.sleep

    async def _invoke(self, make_call: Callable[[], Awaitable[R]]) -> R:
        """Run an LLM call under the per-request timeout, retrying rate limits with backoff."""
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._request_timeout is None:
                    return await make_call()
                return await asyncio.wait_for(make_call(), self._request_timeout)
            except TimeoutError as exc:
                if self._request_timeout is None:
                    raise
                raise TimeoutError(
                    f"LLM request exceeded {self._request_timeout:g}s timeout"
                ) from exc
            except Exception as exc:
                if attempt >= self._max_attempts or not _is_rate_limit_error(exc):
                    raise
                delay = min(
                    _retry_after_seconds(exc, _DEFAULT_RETRY_AFTER_SECONDS), self._max_sleep
                )
                await self._sleep(delay)

    async def ask(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
    ) -> LLMResponse[str]:
        """Raw text completion for query generation, filtering, and scoring."""
        messages = _build_messages(prompt, system)
        start = time.monotonic()

        async def _call() -> object:
            return await self._acompletion(
                model=self._model,
                messages=messages,
                temperature=temperature,
                **self._extra_kwargs,
            )

        completion = await self._invoke(_call)
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

        async def _call() -> tuple[T, Any]:
            return await self._instructor.chat.completions.create_with_completion(
                model=self._model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_retries=max_retries,
                **self._extra_kwargs,
            )

        parsed, completion = await self._invoke(_call)
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
    return LLMClient(
        model=model,
        extra_kwargs=extra,
        provider=settings.llm_provider,
        request_timeout=settings.llm_request_timeout_seconds,
        max_attempts=settings.llm_rate_limit_max_attempts,
        max_sleep=settings.llm_rate_limit_max_sleep_seconds,
    )
