"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import os
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import demo, model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_demo_readiness_never_exposes_credentials(monkeypatch):
    from jev_ultrafast import demo

    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-choice-key")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "secret-text-key")
    assert demo.readiness() == {"choices": True, "text": True, "missing": []}
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.delenv("TEXT_MODEL_API_KEY")
    state = demo.readiness()
    assert state["choices"] is False and state["text"] is False
    assert "secret" not in repr(state)


def test_typesafe_config_pins_model_and_validates_transport(monkeypatch):
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_HTTP2", raising=False)
    assert model.typesafe_config() == {"model": "jev-1.13.0", "http2": "auto"}
    monkeypatch.setenv("TYPESAFE_HTTP2", "invalid")
    with pytest.raises(ValueError, match="TYPESAFE_HTTP2"):
        model.typesafe_config()


def test_load_environment_handles_comments_quotes_exports_and_duplicates(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "export TEXT_MODEL_BASE_URL = https://example.test/v1 # local note\n"
        "TEXT_MODEL = 'first model'\nTEXT_MODEL = final-model\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEXT_MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("TEXT_MODEL", "process-model")
    demo.load_environment()
    assert model.text_base_url() == "https://example.test/v1"
    assert os.environ["TEXT_MODEL"] == "process-model"


@pytest.mark.parametrize("base", ["https://example.test/v1 # note", "https://user:pass@example.test/v1", "not a url"])
def test_text_base_url_rejects_malformed_configuration(monkeypatch, base):
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", base)
    with pytest.raises(ValueError, match="TEXT_MODEL_BASE_URL"):
        model.text_base_url()


def test_text_provider_diagnostic_identifies_provider_host_model_and_attempt_without_secret(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "secret-key")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://text.example.test/v1")
    monkeypatch.setenv("TEXT_MODEL", "small-model")
    response = Mock(status_code=503, is_error=True, http_version="HTTP/2", headers={})
    monkeypatch.setattr(model.CLIENT, "post", Mock(return_value=response))
    monkeypatch.setattr(model.time, "sleep", Mock())
    with pytest.raises(model.ModelRequestError) as error:
        model.field_text({"goal": "Find a flight"})
    diagnostic = error.value.diagnostic
    assert {key: diagnostic[key] for key in ("provider", "endpoint_host", "model", "attempt")} == {
        "provider": "text helper",
        "endpoint_host": "text.example.test",
        "model": "small-model",
        "attempt": 3,
    }
    assert "secret-key" not in str(error.value)
    assert "secret-key" not in repr(diagnostic)


def test_decision_request_retries_a_502(monkeypatch):
    response = Mock(status_code=502, is_error=True)
    success = Mock(status_code=200, is_error=False)
    success.json.return_value = {"ok": True}
    monkeypatch.setattr(model.CLIENT, "post", Mock(side_effect=[response, success]))
    sleep = Mock()
    monkeypatch.setattr(model.time, "sleep", sleep)
    assert model.post_json("https://example.test", "key", {}, retry_statuses=frozenset({502})) == {"ok": True}
    assert model.CLIENT.post.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_text_request_does_not_retry_a_502(monkeypatch):
    response = Mock(status_code=502, is_error=True)
    monkeypatch.setattr(model.CLIENT, "post", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="HTTP 502"):
        model.post_json("https://example.test", "key", {})
    model.CLIENT.post.assert_called_once()


def test_missing_typesafe_credential_stops_before_model_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    post = Mock()
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        model.choose(page(), "Find a book", [])
    post.assert_not_called()


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body, **_kwargs):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body, **_kwargs):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body, **_kwargs):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
        "evidence": [],
        "evidence_truncated": False,
        "report": None,
        "report_error": None,
        "stop_reason": None,
    }
    return a


def test_prediction_failure_never_mutates_or_records(runner, monkeypatch):
    runner.state["decision"] = None
    runner.state["status"] = "ready"
    monkeypatch.setattr(loop, "choose", Mock(side_effect=ValueError("Model unavailable")))
    with pytest.raises(ValueError, match="Model unavailable"):
        runner.command("tick")
    runner.state["browser"].act.assert_not_called()
    assert runner.state["history"] == []
    assert runner.state["decision"] is None


def test_text_helper_failure_preserves_decision_without_mutating(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("TEXT_MODEL_API_KEY missing")))
    original = runner.state["decision"]
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["history"] == []
    assert runner.state["decision"] is original
    assert runner.state["status"] == "predicted"


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_text_helper_rejects_empty_completion_list(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": []}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


@pytest.mark.parametrize("url", [
    "file:///tmp/x", "javascript:alert(1)", "data:text/plain,x",
    "https://user:pass@example.test/", "https://example.test:bad/",
    "https://example.test/a b", "https://example.test/\nnext",
])
def test_start_url_validation_rejects_unsafe_or_malformed_urls(url):
    with pytest.raises(ValueError, match="Start URL"):
        model.validate_browser_url(url)


def test_explicit_start_url_wins_without_routing_or_closing_old_session(monkeypatch):
    old = Mock()
    candidate = Mock()
    candidate.state = {"text_calls": []}
    candidate.snapshot.return_value = {"page": None, "status": "ready", "history": [], "decision": None}
    make_agent = Mock(return_value=candidate)
    route = Mock(side_effect=AssertionError("explicit URL must not route"))
    monkeypatch.setattr(demo, "AGENT", old)
    monkeypatch.setattr(demo, "Agent", make_agent)
    monkeypatch.setattr(demo, "choose_start_url", route)
    demo.command("reset", {"scenario": "custom", "goal": "Find the docs", "url": "https://example.test/start"})
    make_agent.assert_called_once_with("https://example.test/start", "Find the docs", screenshots=True, record_dir=None)
    route.assert_not_called()
    old.close.assert_called_once()
    assert demo.AGENT is candidate


def test_generic_task_routes_with_text_model_and_routing_failure_preserves_session(monkeypatch):
    old = Mock()
    monkeypatch.setattr(demo, "AGENT", old)
    route = Mock(return_value=("https://research.example/", {"model": "text", "purpose": "automatic site selection"}))
    candidate = Mock(state={"text_calls": []})
    candidate.snapshot.return_value = {"page": None, "status": "ready", "history": [], "decision": None}
    monkeypatch.setattr(demo, "choose_start_url", route)
    monkeypatch.setattr(demo, "Agent", Mock(return_value=candidate))
    demo.command("reset", {"scenario": "custom", "goal": "Research browser agents", "url": ""})
    route.assert_called_once_with("Research browser agents")
    assert candidate.state["start_url"] == "https://research.example/"
    assert candidate.state["text_calls"][0]["purpose"] == "automatic site selection"

    current = demo.AGENT
    monkeypatch.setattr(demo, "choose_start_url", Mock(side_effect=ValueError("bad route")))
    with pytest.raises(ValueError, match="bad route"):
        demo.command("reset", {"scenario": "custom", "goal": "Another task", "url": ""})
    assert demo.AGENT is current
    current.close.assert_not_called()


def test_evidence_deduplicates_bounds_and_keeps_changed_same_url(runner, monkeypatch):
    runner.state["evidence"] = []
    first = {"url": "https://example.test/results", "title": "Results", "text": "one", "links": []}
    runner._collect_evidence(first)
    runner._collect_evidence(first)
    runner._collect_evidence({**first, "text": "two"})
    assert [item["text"] for item in runner.state["evidence"]] == ["one", "two"]
    monkeypatch.setattr(loop, "MAX_EVIDENCE_PAGES", 2)
    runner._collect_evidence({**first, "text": "three"})
    assert len(runner.state["evidence"]) == 2
    assert runner.state["evidence_truncated"] is True


def test_final_report_rejects_unobserved_source(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": json.dumps({
        "answer": "Found one item.", "sources": ["https://invented.example/"], "limitations": []
    })}}]}))
    evidence = [{"url": "https://observed.example/", "title": "Observed", "text": "one item", "links": []}]
    with pytest.raises(ValueError, match="ungrounded"):
        model.final_report("Find items", evidence, "done", False)


def test_report_retry_preserves_evidence_and_never_replays_browser_action(runner, monkeypatch):
    runner.state["evidence"] = [{"url": "https://example.test/", "title": "Page", "text": "fact", "links": []}]
    report = Mock(side_effect=[
        ValueError("temporary"),
        ({"answer": "fact", "sources": ["https://example.test/"], "limitations": ["partial"]}, {"model": "text"}),
    ])
    monkeypatch.setattr(loop, "final_report", report)
    with pytest.raises(ValueError, match="temporary"):
        runner.command("report")
    result = runner.command("report")
    assert result["report"]["answer"] == "fact"
    assert result["evidence"][0]["text"] == "fact"
    runner.state["browser"].act.assert_not_called()
    assert report.call_count == 2
