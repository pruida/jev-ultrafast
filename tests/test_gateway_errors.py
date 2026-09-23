"""Handle upstream errors wrapped in successful gateway HTTP responses."""
from unittest.mock import Mock

import httpx
import pytest

from jev_ultrafast import model


@pytest.mark.parametrize("code", [503, "503", 429])
def test_embedded_transient_error_retries_model_request(monkeypatch, code):
    client = Mock()
    client.post.side_effect = [
        httpx.Response(200, json={"error": {"code": code}}),
        httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]}),
    ]
    monkeypatch.setattr(model.time, "sleep", Mock())
    result = model.post_json("https://example.test/chat/completions", "test", {}, client=client)
    assert "choices" in result
    assert client.post.call_count == 2


@pytest.mark.parametrize("code", [401, 400, "unsupported_model", None, []])
def test_embedded_nontransient_error_is_safe_and_not_retried(code):
    client = Mock()
    client.post.return_value = httpx.Response(200, json={"error": {
        "code": code, "message": "private payload secret-key", "metadata": {"raw": "private"},
    }})
    with pytest.raises(model.ModelRequestError) as caught:
        model.post_json("https://example.test/chat/completions", "secret-key", {}, client=client)
    assert "error in HTTP 200" in str(caught.value)
    assert "private" not in str(caught.value) + repr(caught.value.diagnostic)
    assert "secret-key" not in str(caught.value) + repr(caught.value.diagnostic)
    client.post.assert_called_once()


def test_embedded_error_retry_is_bounded(monkeypatch):
    client = Mock()
    client.post.return_value = httpx.Response(200, json={"error": {"code": 503}})
    monkeypatch.setattr(model.time, "sleep", Mock())
    with pytest.raises(model.ModelRequestError) as caught:
        model.post_json("https://example.test/chat/completions", "test", {}, client=client)
    assert client.post.call_count == 3
    assert caught.value.diagnostic["attempt"] == 3
