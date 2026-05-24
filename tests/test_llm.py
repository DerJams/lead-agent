"""Tests for the LLM adapter."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from lead_agent.llm import (
    CallStats,
    LLMClient,
    LLMResponse,
    _build_messages,
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
