"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import time
from pathlib import Path

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text, final_report
from .questions import MAX_STEPS

MAX_EVIDENCE_PAGES = 20
MAX_EVIDENCE_CHARS = 60000


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            evidence=[],
            evidence_truncated=False,
            report=None,
            report_error=None,
            stop_reason=None,
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        self._collect_evidence(page)
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        public = {k: v for k, v in self.state.items() if k != "browser"}
        public["evidence"] = [{k: v for k, v in item.items() if k != "_signature"} for item in self.state["evidence"]]
        return {
            **public,
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def _collect_evidence(self, page):
        state = self.state
        text = page.get("text", "")
        links = page.get("links", [])
        item = {"url": page["url"], "title": page.get("title", ""), "text": text, "links": links}
        signature = (item["url"], item["title"], item["text"], tuple((x.get("label"), x.get("href")) for x in links))
        if any(e.get("_signature") == signature for e in state["evidence"]):
            return
        used = sum(len(e["text"]) + sum(len(x.get("label", "")) + len(x.get("href", "")) for x in e["links"])
                   for e in state["evidence"])
        if len(state["evidence"]) >= MAX_EVIDENCE_PAGES or used >= MAX_EVIDENCE_CHARS:
            state["evidence_truncated"] = True
            return
        remaining = MAX_EVIDENCE_CHARS - used
        item["text"] = item["text"][:remaining]
        remaining -= len(item["text"])
        kept = []
        for link in links:
            cost = len(link.get("label", "")) + len(link.get("href", ""))
            if cost > remaining:
                state["evidence_truncated"] = True
                break
            kept.append(link)
            remaining -= cost
        item["links"] = kept
        item["_signature"] = (
            item["url"], item["title"], item["text"],
            tuple((x.get("label"), x.get("href")) for x in kept),
        )
        state["evidence"].append(item)

    def generate_report(self):
        state = self.state
        evidence = [{k: v for k, v in item.items() if k != "_signature"} for item in state["evidence"]]
        report, helper = final_report(
            state["goal"], evidence, state["stop_reason"] or state["status"], state["evidence_truncated"]
        )
        state["report"] = report
        state["report_error"] = None
        state["text_calls"].append(helper)
        return self.snapshot()

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                self._collect_evidence(state["page"])
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            state["decision"] = choose(state["page"], state["goal"], state["history"])
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["stop_reason"] = selected.lower()
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    try:
                        text, helper = field_text(context)
                    except (ValueError, RuntimeError):
                        # No browser input occurred, so keep the inspected decision available to retry.
                        state["decision"] = decision
                        state["status"] = "predicted"
                        raise
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"].get(selected),
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            self._collect_evidence(state["page"])
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            repeated = state["history"][-3:]
            stuck = len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
            state["status"] = "blocked" if stuck else "ready"
            if stuck:
                state["stop_reason"] = "no_progress"
        elif name == "report":
            return self.generate_report()
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
