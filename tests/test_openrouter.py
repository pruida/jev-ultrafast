"""Offline OpenRouter decision contracts; no provider calls."""
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.demo import readiness


def context():
    return {"url": "https://example.test", "title": "Search", "text": "Search", "actions": [
        {"id": "fill", "node": 1, "kind": "fill", "label": "Query", "value": ""},
        {"id": "click", "node": 2, "kind": "click", "label": "Go", "value": ""},
        {"id": "select", "node": 3, "kind": "select", "label": "Option", "value": "x"},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("DECISION_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("TYPESAFE_API_KEY", "different-secret")


def response(operation, target):
    operations = ("CLICK", "TYPE_TEXT", "SELECT", "WAIT", "DONE", "BLOCKED")
    answers = {
        "operation": {
            "choice": operation,
            "confidence": 0.9,
            "probabilities": {key: 1.0 if key == operation else 0.0 for key in operations},
        },
    }
    if target is not None:
        choices = {"CLICK": ("2",), "TYPE_TEXT": ("1",), "SELECT": ("3:1",)}.get(operation)
        if choices:
            answers[operation.lower() + "_target"] = {
                "choice": target,
                "confidence": 0.8,
                "probabilities": {key: 1.0 if key == target else 0.0 for key in choices},
            }
    return {"answers": answers}


@pytest.mark.parametrize("operation,target,selected", [
    ("CLICK", "2", "click"), ("TYPE_TEXT", "1", "fill"), ("SELECT", "3:1", "select"),
    ("WAIT", None, "wait"), ("DONE", None, "DONE"), ("BLOCKED", None, "BLOCKED"),
])
def test_valid_mapping(configured, monkeypatch, operation, target, selected):
    post = Mock(return_value=response(operation, target))
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(context(), "Search", [])
    assert d["choice"] == selected
    assert d["target"] == target
    assert d["probabilities"] == {selected: 1.0}
    assert d["confidence"] == 0.9
    url, key, body = post.call_args.args
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert key == "router-secret"
    assert body["model"] == "test/model"
    assert {"state", "questions"} <= body.keys()
    assert "messages" not in body and "response_format" not in body and "provider" not in body


@pytest.mark.parametrize("base", [
    "https://decision-gateway.example/v1/",
    " https://decision-gateway.example/v1/// ",
])
def test_custom_gateway_base_url(configured, monkeypatch, base):
    monkeypatch.setenv("OPENROUTER_BASE_URL", base)
    post = Mock(return_value=response("CLICK", "2"))
    monkeypatch.setattr(model, "post_json", post)
    model.choose(context(), "Search", [])
    url, key, _ = post.call_args.args
    assert url == "https://decision-gateway.example/v1/alpha/decisions"
    assert key == "router-secret"


@pytest.mark.parametrize("base", [
    "", "   ", "gateway.example/v1", "ftp://gateway.example/v1",
    "https://gateway.example/v1?key=value", "https://gateway.example/v1#fragment",
])
def test_invalid_gateway_base_stops_before_request(configured, monkeypatch, base):
    monkeypatch.setenv("OPENROUTER_BASE_URL", base)
    post = Mock()
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="OPENROUTER_BASE_URL"):
        model.choose(context(), "Search", [])
    post.assert_not_called()


def test_invalid_gateway_base_does_not_affect_typesafe(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "not-a-url")
    assert model.decision_config()["provider"] == "typesafe"


@pytest.mark.parametrize("operation,target", [
    ("CLICK", "1"), ("TYPE_TEXT", "2"), ("CLICK", "unknown"),
    ("CLICK", None), ("SCRIPT", None), ("CLICK", 2),
])
def test_invalid_pair(configured, monkeypatch, operation, target):
    monkeypatch.setattr(model, "post_json", Mock(return_value=response(operation, target)))
    with pytest.raises(ValueError, match="Invalid OpenRouter"):
        model.choose(context(), "Search", [])


@pytest.mark.parametrize("mutation", ["missing_answers", "missing_operation", "missing_target"])
def test_invalid_response(configured, monkeypatch, mutation):
    result = response("CLICK", "2")
    if mutation == "missing_answers":
        result = {}
    elif mutation == "missing_operation":
        result["answers"] = {}
    else:
        result["answers"].pop("click_target")
    monkeypatch.setattr(model, "post_json", Mock(return_value=result))
    with pytest.raises(ValueError, match="Invalid OpenRouter"):
        model.choose(context(), "Search", [])


@pytest.mark.parametrize("result", [
    None, [], {"answers": None}, {"answers": []},
    {"answers": {"operation": None}},
    {"answers": {"operation": {"probabilities": []}}},
])
def test_malformed_containers(configured, monkeypatch, result):
    monkeypatch.setattr(model, "post_json", Mock(return_value=result))
    with pytest.raises(ValueError, match="Invalid OpenRouter"):
        model.choose(context(), "Search", [])


@pytest.mark.parametrize("operation,target,selected", [
    ("CLICK", "2", "click"), ("TYPE_TEXT", "1", "fill"),
    ("SELECT", "3:1", "select"), ("WAIT", None, "wait"),
])
def test_prediction_executes_and_records_history(configured, monkeypatch, operation, target, selected):
    page = {**context(), "fingerprint": "before"}
    browser = Mock()
    browser.observe.side_effect = [page, {**page, "fingerprint": "after"}]
    browser.fresh.return_value = True
    monkeypatch.setattr(loop, "Browser", Mock(return_value=browser))
    monkeypatch.setattr(model, "post_json", Mock(return_value=response(operation, target)))
    monkeypatch.setattr(loop, "field_text", Mock(return_value=("query", {"model": "helper", "latency_ms": 1})))
    agent = loop.Agent(page["url"], "Search")
    agent.command("predict")
    snapshot = agent.command("act", {"fingerprint": "before"})
    browser.act.assert_called_once()
    assert snapshot["status"] == "ready"
    assert snapshot["decision"] is None
    assert len(snapshot["history"]) == 1
    entry = snapshot["history"][0]
    assert entry["choice"] == selected
    assert entry["target"] == target
    assert entry["probability"] == 1.0
    assert entry["page_changed"] is True
    with pytest.raises(ValueError, match="Observe and choose"):
        agent.command("act", {"fingerprint": "before"})
    browser.act.assert_called_once()


def test_missing_config(configured, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert readiness()["choices"] is False
    post = Mock()
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        model.choose(context(), "Search", [])
    post.assert_not_called()
    monkeypatch.delenv("OPENROUTER_MODEL")
    assert model.decision_config()["model"] == "~typesafe/jev-latest"
