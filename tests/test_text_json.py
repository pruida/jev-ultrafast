"""Safe text-helper envelope diagnostics and complete fenced JSON support."""
from unittest.mock import Mock

import pytest

from jev_ultrafast import model


@pytest.mark.parametrize("content", [
    '{"url":"https://example.test/"}',
    '```json\n{"url":"https://example.test/"}\n```',
])
def test_start_url_accepts_complete_json(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]}))
    assert model.choose_start_url("Find documents")[0] == "https://example.test/"


@pytest.mark.parametrize("completion,reason", [
    ({"finish_reason": "length", "message": {"content": '{"url":'}}, "truncated"),
    ({"message": {"content": None}}, "empty or non-text"),
    ({"message": {"refusal": "private details"}}, "refused"),
    ({"message": {"content": 'Explanation {"url":"https://example.test/"}'}}, "not a JSON object"),
    ({"message": {"content": '[]'}}, "not a JSON object"),
])
def test_start_url_failure_is_specific_without_raw_content(monkeypatch, completion, reason):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "secret-key")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [completion]}))
    with pytest.raises(ValueError, match=reason) as caught:
        model.choose_start_url("private task")
    assert "secret-key" not in str(caught.value)
    assert "private" not in str(caught.value)
