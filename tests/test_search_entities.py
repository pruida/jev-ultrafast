"""Keep full autocomplete entities and submitted search context in decisions."""
from unittest.mock import Mock

from jev_ultrafast import model
from jev_ultrafast.questions import FINAL_REPORT, NEXT_ACTION


def test_full_entity_and_submitted_url_reach_target_question(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    page = {
        "url": "https://example.test/?inventor=林晓&q=明",
        "title": "Search", "text": "Results",
        "actions": [{"id": "e1", "node": 1, "kind": "click", "role": "option",
                     "label": "Inventor: 林晓; keyword: 明", "autocomplete_value": "林晓明"}],
    }
    post = Mock(return_value={"model": "test", "answers": {
        "operation": {"choice": "CLICK", "confidence": 1,
                      "probabilities": {"CLICK": 1, "DONE": 0, "BLOCKED": 0}},
        "click_target": {"choice": "1", "confidence": 1, "probabilities": {"1": 1}},
    }})
    monkeypatch.setattr(model, "post_typesafe_json", post)
    model.choose(page, "Find inventions by 林晓明", [{"url": page["url"], "text": "林晓明"}])
    body = post.call_args.args[2]
    assert body["questions"]["click_target"]["criteria"]["1"]["autocomplete_value"] == "林晓明"
    assert body["state"]["recent_actions"][0]["url"] == page["url"]
    assert "complete intended entity" in NEXT_ACTION
    assert "verify effective filters" in NEXT_ACTION
    assert "partial-name match is not evidence" in FINAL_REPORT
