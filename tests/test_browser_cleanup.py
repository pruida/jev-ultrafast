"""Offline cleanup contracts for owned Chrome targets."""
from unittest.mock import Mock

import pytest

from jev_ultrafast import browser, demo
from jev_ultrafast.agent import Agent


def closed_target():
    return RuntimeError({"code": -32602, "message": "No target with given id found"})


def test_already_closed_target_cleanup_is_idempotent(monkeypatch):
    tab = browser.Browser.__new__(browser.Browser)
    tab.target = "old-target"
    cdp = Mock(side_effect=closed_target())
    monkeypatch.setattr(browser, "cdp", cdp)
    tab.close()
    tab.close()
    assert tab.target is None
    cdp.assert_called_once_with("Target.closeTarget", targetId="old-target")


@pytest.mark.parametrize("detail", [
    {"code": -32602, "message": "Invalid parameters"},
    {"code": -32000, "message": "No target with given id found"},
    "daemon unavailable",
])
def test_other_cleanup_errors_propagate(monkeypatch, detail):
    tab = browser.Browser.__new__(browser.Browser)
    tab.target = "old-target"
    error = RuntimeError(detail)
    monkeypatch.setattr(browser, "cdp", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        tab.close()
    assert raised.value is error
    assert tab.target == "old-target"


def test_reset_succeeds_when_old_chrome_target_is_gone(monkeypatch):
    old = Agent.__new__(Agent)
    old.browser = browser.Browser.__new__(browser.Browser)
    old.browser.target = "old-target"
    monkeypatch.setattr(browser, "cdp", Mock(side_effect=closed_target()))
    candidate = Mock(state={"text_calls": []})
    candidate.snapshot.return_value = {"page": None, "status": "ready", "history": [], "decision": None}
    monkeypatch.setattr(demo, "AGENT", old)
    monkeypatch.setattr(demo, "Agent", Mock(return_value=candidate))
    result = demo.command("reset", {
        "scenario": "custom", "goal": "Find patents", "url": "https://patents.google.com/",
    })
    assert demo.AGENT is candidate
    assert result["status"] == "ready"
    assert old.browser.target is None
