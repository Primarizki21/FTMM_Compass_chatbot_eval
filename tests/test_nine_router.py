import pytest

from lib import llm


class FakeResponse:
    """Tiruan respons httpx untuk stream (iter_lines/read) dan non-stream (json)."""

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


def set_router(monkeypatch, host="https://tunnel.example/v1", key="sk-uji"):
    monkeypatch.setenv("NINE_ROUTER_HOST", host)
    monkeypatch.setenv("NINE_ROUTER_API_KEY", key)


def test_nine_router_stream_collects_tokens_and_usage(monkeypatch):
    set_router(monkeypatch)
    captured = {}
    lines = [
        'data: {"choices":[{"delta":{"content":"Halo "}}]}',
        "data: bukan-json",
        'data: {"choices":[{"delta":{"content":"dunia"}}],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ]

    def fake_stream(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse(lines)

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    stats = {}
    answer = "".join(
        llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.NINE_ROUTER, "ag/gemini-x"), 0, stats)
    )

    assert answer == "Halo dunia"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://tunnel.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-uji"
    assert captured["json"]["model"] == "ag/gemini-x"
    assert captured["json"]["stream"] is True
    assert stats["prompt_tokens"] == 7
    assert stats["answer_tokens"] == 2
    assert stats["generation_ms"] >= stats["ttft_ms"]


def test_nine_router_rejects_http_error_with_body(monkeypatch):
    set_router(monkeypatch)
    response = FakeResponse(status_code=401, body=b'{"error":"invalid key"}')
    monkeypatch.setattr(llm.httpx, "stream", lambda *args, **kwargs: response)

    with pytest.raises(llm.LLMError, match="401"):
        list(llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.NINE_ROUTER, "m"), 0, {}))


def test_nine_router_requires_host_key_and_model(monkeypatch):
    for name in ("NINE_ROUTER_HOST", "NINE_ROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert "NINE_ROUTER_HOST" in llm.nine_router_setup_problem("m")
    with pytest.raises(llm.LLMError, match="NINE_ROUTER_HOST"):
        list(llm.stream_chat([{"role": "user", "content": "halo"}], llm.LLMConfig(llm.NINE_ROUTER, "m"), 0, {}))

    monkeypatch.setenv("NINE_ROUTER_HOST", "https://tunnel.example")
    assert "NINE_ROUTER_API_KEY" in llm.nine_router_setup_problem("m")

    monkeypatch.setenv("NINE_ROUTER_API_KEY", "sk-uji")
    assert "Model" in llm.nine_router_setup_problem("  ")
    assert llm.nine_router_setup_problem("ag/gemini-x") is None


def test_nine_router_reformulation_uses_low_temperature(monkeypatch):
    set_router(monkeypatch, host="https://tunnel.example")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(payload={"choices": [{"message": {"content": " Algoritma Pemrograman\n"}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    text = llm.reformulate_query("belajar koding dasar", llm.LLMConfig(llm.NINE_ROUTER, "ag/gemini-x"))

    assert text == "Algoritma Pemrograman"
    assert captured["url"] == "https://tunnel.example/v1/chat/completions"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["max_tokens"] == 30
