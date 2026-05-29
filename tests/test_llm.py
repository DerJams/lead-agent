"""Tests for the LLM adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from pydantic import BaseModel

from lead_agent.llm import (
    CallStats,
    LLMClient,
    LLMResponse,
    _build_messages,
    _is_rate_limit_error,
    _parse_retry_after_message,
    _retry_after_seconds,
    get_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_completion(
    content: str = "result text",
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Minimal fake litellm ModelResponse."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    completion.model = model
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    return completion


def make_ask_client(
    content: str = "result text",
    provider: str = "ollama",
) -> tuple[LLMClient, AsyncMock]:
    """LLMClient wired to a mock acompletion; returns (client, mock)."""
    completion = make_completion(content=content)
    mock_ac = AsyncMock(return_value=completion)
    client = LLMClient(
        model=f"{provider}/test-model",
        extra_kwargs={},
        provider=provider,
        _acompletion=mock_ac,
    )
    return client, mock_ac


def make_extract_client(
    parsed: BaseModel,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    provider: str = "ollama",
) -> tuple[LLMClient, MagicMock]:
    """LLMClient wired to a mock instructor client; returns (client, mock_instructor)."""
    completion = make_completion(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    mock_instructor = MagicMock()
    mock_instructor.chat.completions.create_with_completion = AsyncMock(
        return_value=(parsed, completion)
    )
    client = LLMClient(
        model=f"{provider}/test-model",
        extra_kwargs={},
        provider=provider,
        _instructor_client=mock_instructor,
    )
    return client, mock_instructor


# ---------------------------------------------------------------------------
# LLMResponse and CallStats
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_content_and_stats_attributes(self) -> None:
        stats = CallStats(
            model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.0, duration_ms=10
        )
        resp = LLMResponse(content="hello", stats=stats)
        assert resp.content == "hello"
        assert resp.stats is stats

    def test_generic_over_dataclass(self) -> None:
        @dataclass
        class Dummy:
            value: int = 42

        stats = CallStats(
            model="m", prompt_tokens=0, completion_tokens=0, cost_usd=0.0, duration_ms=0
        )
        resp: LLMResponse[Dummy] = LLMResponse(content=Dummy(), stats=stats)
        assert resp.content.value == 42

    def test_generic_over_pydantic_model(self) -> None:
        class MyModel(BaseModel):
            name: str

        stats = CallStats(
            model="m", prompt_tokens=0, completion_tokens=0, cost_usd=0.0, duration_ms=0
        )
        resp: LLMResponse[MyModel] = LLMResponse(content=MyModel(name="test"), stats=stats)
        assert resp.content.name == "test"


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_prompt_only(self) -> None:
        msgs = _build_messages("hello", "")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_with_system_prepends_system_message(self) -> None:
        msgs = _build_messages("hello", "be terse")
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "hello"}
        assert len(msgs) == 2

    def test_empty_system_omitted(self) -> None:
        msgs = _build_messages("q", "")
        assert all(m["role"] != "system" for m in msgs)


# ---------------------------------------------------------------------------
# get_client() configuration
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_ollama_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        client = get_client()
        assert client._model == "ollama/llama3.1:8b"

    def test_ollama_api_base_in_extra_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        client = get_client()
        assert client._extra_kwargs.get("api_base") == "http://localhost:11434"

    def test_ollama_provider_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        client = get_client()
        assert client._provider == "ollama"

    def test_groq_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = get_client()
        assert client._model == "groq/llama-3.3-70b-versatile"

    def test_groq_no_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        client = get_client()
        assert "api_base" not in client._extra_kwargs

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        with pytest.raises(ValueError, match="LLM_PROVIDER setting"):
            get_client()


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------

class TestAsk:
    async def test_returns_llm_response_with_string_content(self) -> None:
        client, _ = make_ask_client(content="Paris")
        resp = await client.ask("What is the capital of France?")
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Paris"

    async def test_stats_token_counts(self) -> None:
        client, _ = make_ask_client()
        resp = await client.ask("prompt")
        assert resp.stats.prompt_tokens == 10
        assert resp.stats.completion_tokens == 5

    async def test_stats_duration_non_negative(self) -> None:
        client, _ = make_ask_client()
        resp = await client.ask("prompt")
        assert resp.stats.duration_ms >= 0

    async def test_ollama_cost_is_zero(self) -> None:
        client, _ = make_ask_client(provider="ollama")
        resp = await client.ask("prompt")
        assert resp.stats.cost_usd == 0.0

    async def test_system_prompt_included_in_messages(self) -> None:
        client, mock_ac = make_ask_client()
        await client.ask("question", system="be brief")
        call_messages = mock_ac.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": "be brief"}

    async def test_no_system_prompt_by_default(self) -> None:
        client, mock_ac = make_ask_client()
        await client.ask("question")
        call_messages = mock_ac.call_args.kwargs["messages"]
        assert all(m["role"] != "system" for m in call_messages)

    async def test_temperature_passed_through(self) -> None:
        client, mock_ac = make_ask_client()
        await client.ask("q", temperature=0.7)
        assert mock_ac.call_args.kwargs["temperature"] == 0.7

    async def test_model_string_passed_through(self) -> None:
        client, mock_ac = make_ask_client()
        await client.ask("q")
        assert mock_ac.call_args.kwargs["model"] == client._model


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

class TestExtract:
    async def test_returns_llm_response_with_pydantic_model(self) -> None:
        class FirmName(BaseModel):
            name: str

        parsed = FirmName(name="Acme Law")
        client, _ = make_extract_client(parsed=parsed)
        resp = await client.extract("Extract the firm name", FirmName)
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.content, FirmName)
        assert resp.content.name == "Acme Law"

    async def test_stats_token_counts(self) -> None:
        class Simple(BaseModel):
            value: str

        client, _ = make_extract_client(
            parsed=Simple(value="x"), prompt_tokens=20, completion_tokens=8
        )
        resp = await client.extract("prompt", Simple)
        assert resp.stats.prompt_tokens == 20
        assert resp.stats.completion_tokens == 8

    async def test_max_retries_passed_to_instructor(self) -> None:
        class Simple(BaseModel):
            value: str

        client, mock_instructor = make_extract_client(parsed=Simple(value="x"))
        await client.extract("prompt", Simple, max_retries=3)
        kwargs = mock_instructor.chat.completions.create_with_completion.call_args.kwargs
        assert kwargs["max_retries"] == 3

    async def test_response_model_passed_to_instructor(self) -> None:
        class Simple(BaseModel):
            value: str

        client, mock_instructor = make_extract_client(parsed=Simple(value="x"))
        await client.extract("prompt", Simple)
        kwargs = mock_instructor.chat.completions.create_with_completion.call_args.kwargs
        assert kwargs["response_model"] is Simple

    async def test_system_prompt_included(self) -> None:
        class Simple(BaseModel):
            value: str

        client, mock_instructor = make_extract_client(parsed=Simple(value="x"))
        await client.extract("prompt", Simple, system="be precise")
        kwargs = mock_instructor.chat.completions.create_with_completion.call_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "be precise"}


# ---------------------------------------------------------------------------
# Rate-limit / retry-after helpers (pure functions)
# ---------------------------------------------------------------------------

class _NamedRateLimitError(Exception):
    """Carries a fake response so header-based retry-after parsing can be exercised."""

    def __init__(self, message: str, retry_after_header: str | None = None) -> None:
        super().__init__(message)
        if retry_after_header is not None:
            self.response = SimpleNamespace(headers={"retry-after": retry_after_header})


def make_groq_rate_limit_error(seconds: float = 5.5) -> litellm.RateLimitError:
    """A real litellm.RateLimitError shaped like Groq's free-tier 429."""
    return litellm.RateLimitError(
        message=(
            "Rate limit reached for model `llama-3.3-70b-versatile` ... on tokens per "
            f"minute (TPM): Limit 12000. Please try again in {seconds}s."
        ),
        llm_provider="groq",
        model="llama-3.3-70b-versatile",
    )


class RecordingSleep:
    """Async sleep stand-in that records requested delays without actually waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


class TestRateLimitDetection:
    def test_real_litellm_rate_limit_error_detected(self) -> None:
        assert _is_rate_limit_error(make_groq_rate_limit_error()) is True

    def test_wrapped_cause_detected(self) -> None:
        wrapper = RuntimeError("instructor gave up")
        wrapper.__cause__ = make_groq_rate_limit_error()
        assert _is_rate_limit_error(wrapper) is True

    def test_string_heuristic_detected(self) -> None:
        assert _is_rate_limit_error(Exception("got rate_limit_exceeded from api")) is True

    def test_unrelated_error_not_detected(self) -> None:
        assert _is_rate_limit_error(ValueError("bad json")) is False


class TestParseRetryAfter:
    def test_seconds_with_decimal(self) -> None:
        assert _parse_retry_after_message("Please try again in 8.855s.") == pytest.approx(8.855)

    def test_minutes_and_seconds(self) -> None:
        assert _parse_retry_after_message("try again in 1m30s") == pytest.approx(90.0)

    def test_no_match_returns_none(self) -> None:
        assert _parse_retry_after_message("no hint here") is None

    def test_message_source(self) -> None:
        err = make_groq_rate_limit_error(seconds=12.0)
        assert _retry_after_seconds(err, default=99.0) == pytest.approx(12.0)

    def test_header_source(self) -> None:
        err = _NamedRateLimitError("rate limited", retry_after_header="7")
        assert _retry_after_seconds(err, default=99.0) == pytest.approx(7.0)

    def test_falls_back_to_default(self) -> None:
        err = _NamedRateLimitError("rate limited, no hint")
        assert _retry_after_seconds(err, default=42.0) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Retry + timeout behavior on the client
# ---------------------------------------------------------------------------

def make_retry_client(
    *,
    acompletion: object | None = None,
    instructor_client: object | None = None,
    request_timeout: float | None = 60.0,
    max_attempts: int = 3,
    max_sleep: float = 60.0,
) -> tuple[LLMClient, RecordingSleep]:
    sleeper = RecordingSleep()
    client = LLMClient(
        model="groq/test-model",
        extra_kwargs={},
        provider="groq",
        _acompletion=acompletion,
        _instructor_client=instructor_client,
        request_timeout=request_timeout,
        max_attempts=max_attempts,
        max_sleep=max_sleep,
        _sleep=sleeper,
    )
    return client, sleeper


class TestRetryAfterHonored:
    async def test_extract_sleeps_retry_after_then_succeeds(self) -> None:
        class Simple(BaseModel):
            value: str

        parsed = Simple(value="ok")
        completion = make_completion()
        mock_create = AsyncMock(
            side_effect=[make_groq_rate_limit_error(seconds=6.0), (parsed, completion)]
        )
        instructor_client = MagicMock()
        instructor_client.chat.completions.create_with_completion = mock_create
        client, sleeper = make_retry_client(instructor_client=instructor_client)

        resp = await client.extract("prompt", Simple)

        assert resp.content is parsed
        assert mock_create.call_count == 2  # one 429, one success
        assert sleeper.calls == [pytest.approx(6.0)]  # honored the retry-after

    async def test_ask_sleeps_retry_after_then_succeeds(self) -> None:
        mock_ac = AsyncMock(
            side_effect=[make_groq_rate_limit_error(seconds=3.0), make_completion(content="hi")]
        )
        client, sleeper = make_retry_client(acompletion=mock_ac)

        resp = await client.ask("q")

        assert resp.content == "hi"
        assert mock_ac.call_count == 2
        assert sleeper.calls == [pytest.approx(3.0)]

    async def test_retry_after_capped_at_max_sleep(self) -> None:
        mock_ac = AsyncMock(
            side_effect=[make_groq_rate_limit_error(seconds=600.0), make_completion()]
        )
        client, sleeper = make_retry_client(acompletion=mock_ac, max_sleep=30.0)

        await client.ask("q")

        assert sleeper.calls == [pytest.approx(30.0)]  # clamped


class TestRetriesExhausted:
    async def test_extract_raises_after_max_attempts(self) -> None:
        class Simple(BaseModel):
            value: str

        mock_create = AsyncMock(side_effect=make_groq_rate_limit_error(seconds=2.0))
        instructor_client = MagicMock()
        instructor_client.chat.completions.create_with_completion = mock_create
        client, sleeper = make_retry_client(instructor_client=instructor_client, max_attempts=3)

        with pytest.raises(litellm.RateLimitError):
            await client.extract("prompt", Simple)

        assert mock_create.call_count == 3  # all attempts used
        assert sleeper.calls == [pytest.approx(2.0), pytest.approx(2.0)]  # one fewer than attempts

    async def test_ask_raises_after_max_attempts(self) -> None:
        mock_ac = AsyncMock(side_effect=make_groq_rate_limit_error(seconds=1.0))
        client, sleeper = make_retry_client(acompletion=mock_ac, max_attempts=2)

        with pytest.raises(litellm.RateLimitError):
            await client.ask("q")

        assert mock_ac.call_count == 2
        assert sleeper.calls == [pytest.approx(1.0)]


class TestNoRetryCases:
    async def test_non_rate_limit_error_not_retried(self) -> None:
        mock_ac = AsyncMock(side_effect=ValueError("boom"))
        client, sleeper = make_retry_client(acompletion=mock_ac)

        with pytest.raises(ValueError, match="boom"):
            await client.ask("q")

        assert mock_ac.call_count == 1  # no retry on non-rate-limit errors
        assert sleeper.calls == []

    async def test_success_path_does_not_sleep_or_retry(self) -> None:
        mock_ac = AsyncMock(return_value=make_completion(content="fine"))
        client, sleeper = make_retry_client(acompletion=mock_ac)

        resp = await client.ask("q")

        assert resp.content == "fine"
        assert mock_ac.call_count == 1
        assert sleeper.calls == []


class TestRequestTimeout:
    async def test_timeout_raises_timeout_error(self) -> None:
        async def slow_ac(**kwargs: object) -> object:
            await asyncio.sleep(1.0)
            return make_completion()

        client, sleeper = make_retry_client(acompletion=slow_ac, request_timeout=0.01)

        with pytest.raises(TimeoutError, match="exceeded"):
            await client.ask("q")

        assert sleeper.calls == []  # a timeout is not a rate-limit retry

    async def test_no_timeout_when_disabled(self) -> None:
        mock_ac = AsyncMock(return_value=make_completion(content="ok"))
        client, _ = make_retry_client(acompletion=mock_ac, request_timeout=None)

        resp = await client.ask("q")

        assert resp.content == "ok"
