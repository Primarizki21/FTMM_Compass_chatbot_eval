import pytest

from lib import llm


class FakeResponse:
    def __init__(self, lines=(), status_code=200, body=b"", payload=None):
        self._lines = lines
        self.status_code = status_code
        self._body = body
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def json(self):
        return self._payload


def set_openai(monkeypatch, key="sk-uji", effort="none"):
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(llm, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", effort)


def test_openai_stream_collects_tokens_and_usage(monkeypatch):
    set_openai(monkeypatch)
    captured = {}
    lines = [
        'data: {"choices":[{"delta":{"content":"Halo "}}]}',
        "data: bukan-json",
        'data: {"choices":[{"delta":{"content":"dunia"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}',
        "data: [DONE]",
    ]

    def fake_stream(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse(lines)

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    stats = {}
    answer = "".join(
        llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.OPENAI, "gpt-6-luna"), 0, stats)
    )
    payload = captured["json"]

    assert answer == "Halo dunia"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-uji"
    assert payload["model"] == "gpt-6-luna"
    assert payload["max_completion_tokens"] == llm.CHAT_OPTIONS["num_predict"]
    assert "max_tokens" not in payload
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["reasoning_effort"] == "none"
    assert payload["temperature"] == llm.CHAT_OPTIONS["temperature"]
    assert stats["prompt_tokens"] == 11
    assert stats["answer_tokens"] == 3
    assert stats["generation_ms"] >= stats["ttft_ms"]


def test_openai_drops_sampling_when_reasoning_active(monkeypatch):
    set_openai(monkeypatch, effort="low")
    captured = {}

    def fake_stream(method, url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)

    list(llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.OPENAI, "gpt-6-luna"), 0, {}))

    payload = captured["json"]
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_openai_rejects_http_error_with_body(monkeypatch):
    set_openai(monkeypatch)
    monkeypatch.setattr(
        llm.httpx, "stream", lambda *args, **kwargs: FakeResponse(status_code=401, body=b'{"error":"invalid key"}')
    )

    with pytest.raises(llm.LLMError, match="OpenAI.*401"):
        list(llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.OPENAI, "gpt-6-luna"), 0, {}))


def test_openai_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" in llm.openai_setup_problem("gpt-6-luna")
    with pytest.raises(llm.LLMError, match="OPENAI_API_KEY"):
        list(llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.OPENAI, "gpt-6-luna"), 0, {}))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-uji")
    assert "Model" in llm.openai_setup_problem("  ")
    assert llm.openai_setup_problem("gpt-6-luna") is None


def test_openai_reformulation_uses_low_temperature(monkeypatch):
    set_openai(monkeypatch)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(payload={"choices": [{"message": {"content": " Algoritma Pemrograman\n"}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    text = llm.reformulate_query("belajar koding dasar", llm.LLMConfig(llm.OPENAI, "gpt-6-luna"))

    assert text == "Algoritma Pemrograman"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["max_completion_tokens"] == 30
    assert "stream_options" not in captured["json"]
