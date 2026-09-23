"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import hashlib
import json
import math
import os
import time
from urllib.parse import urlsplit

import httpx

from .questions import FINAL_REPORT, NEXT_ACTION, START_URL, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class ModelRequestError(RuntimeError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def typesafe_config():
    model = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0").strip()
    if not model:
        raise ValueError("TYPESAFE_MODEL cannot be blank.")
    http2 = os.environ.get("TYPESAFE_HTTP2", "auto").strip().lower()
    if http2 not in {"auto", "true", "false"}:
        raise ValueError("TYPESAFE_HTTP2 must be auto, true, or false.")
    return {"model": model, "http2": http2}


def endpoint_host(url, name="TEXT_MODEL_BASE_URL"):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL without credentials, query, or fragment")
    if parsed.query or parsed.fragment or any(char.isspace() for char in url):
        raise ValueError(f"{name} must not contain whitespace, query, or fragment")
    return parsed.netloc


def api_base_url(name, default):
    base = os.environ.get(name, default).strip().rstrip("/")
    endpoint_host(base, name)
    return base


def text_base_url():
    return api_base_url("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")


def openrouter_base_url():
    return api_base_url("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)


def payload_metadata(body):
    encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    return {"payload_bytes": len(encoded), "payload_sha256": hashlib.sha256(encoded).hexdigest()}


def post_json(
    url,
    key,
    body,
    *,
    retry_statuses=frozenset({429, 503, 529}),
    diagnostic=None,
    client=None,
    provider=None,
    model=None,
):
    diagnostic = dict(diagnostic or {})
    parsed = urlsplit(url)
    diagnostic.update(endpoint_host=parsed.netloc, endpoint_path=parsed.path)
    if provider is not None:
        diagnostic["provider"] = provider
    if model is not None:
        diagnostic["model"] = model
    client = client or CLIENT
    for attempt in range(3):
        started = time.perf_counter()
        try:
            response = client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as error:
            diagnostic.update(attempt=attempt + 1, latency_ms=round((time.perf_counter() - started) * 1000))
            raise ModelRequestError("Model connection failed; no action executed.", diagnostic) from error
        if response.status_code in retry_statuses and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            diagnostic.update(
                attempt=attempt + 1,
                latency_ms=round((time.perf_counter() - started) * 1000),
                status_code=response.status_code,
                http_version=response.http_version,
                response_request_id=(
                    response.headers.get("x-typesafe-request-id") or response.headers.get("x-request-id")
                ),
            )
            raise ModelRequestError(
                f"Model provider returned HTTP {response.status_code}; no action executed.", diagnostic
            )
        result = response.json()
        if isinstance(result, dict) and result.get("error") is not None:
            error = result["error"]
            code = error.get("code") if isinstance(error, dict) else None
            if isinstance(code, str) and code.isdecimal():
                code = int(code)
            if type(code) is int and code in retry_statuses and attempt < 2:
                time.sleep(0.5 * 2**attempt)
                continue
            safe_code = code if type(code) is int else "unknown"
            diagnostic.update(
                attempt=attempt + 1, status_code=response.status_code, provider_error_code=safe_code,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            raise ModelRequestError(
                f"Model provider returned an error in HTTP {response.status_code} "
                f"(provider code: {safe_code}); no action executed.", diagnostic,
            )
        return result
    raise RuntimeError("Model unavailable")


def post_typesafe_json(url, key, body, config):
    metadata = {
        **payload_metadata(body),
        "http2_setting": config["http2"],
    }
    if config["http2"] != "false":
        return post_json(
            url,
            key,
            body,
            retry_statuses=frozenset({429, 502, 503, 529}),
            diagnostic=metadata,
            provider="typesafe",
            model=config["model"],
        )
    with httpx.Client(http2=False, timeout=25) as client:
        return post_json(
            url,
            key,
            body,
            retry_statuses=frozenset({429, 502, 503, 529}),
            diagnostic=metadata,
            client=client,
            provider="typesafe",
            model=config["model"],
        )


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions, *, max_actions=80):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        if action["kind"] in operations and action["node"] not in indices and len(elements) >= max_actions:
            continue
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def decision_config():
    provider = os.environ.get("DECISION_PROVIDER", "typesafe").strip().lower()
    if provider == "typesafe":
        return {**typesafe_config(), "provider": provider, "key_name": "TYPESAFE_API_KEY"}
    if provider != "openrouter":
        raise ValueError("DECISION_PROVIDER must be typesafe or openrouter.")
    model = os.environ.get("OPENROUTER_MODEL", "~typesafe/jev-latest").strip()
    if not model:
        raise ValueError("OPENROUTER_MODEL cannot be blank.")
    return {
        "provider": provider,
        "model": model,
        "key_name": "OPENROUTER_API_KEY",
        "base_url": openrouter_base_url(),
    }


def choose_openrouter(body, targets, controls, config, key):
    request = {**body, "model": config["model"]}
    started = time.perf_counter()
    result = post_json(
        config["base_url"] + "/alpha/decisions", key, request,
        provider="openrouter", model=config["model"], diagnostic=payload_metadata(request),
        retry_statuses=frozenset({429, 502, 503, 529}),
    )
    decision = parse_decision(result, targets, controls, config, provider="openrouter", request=request)
    decision["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return decision


def parse_decision(result, targets, controls, config, *, provider, request):
    try:
        operations = request["questions"]["operation"]["criteria"]
        operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
        operation = operation_answer["choice"]
        if operation in targets:
            target_answer = validate_choice(
                result["answers"].get(operation.lower() + "_target", {}), targets[operation]
            )
            selected = targets[operation][target_answer["choice"]]["id"]
            probabilities = {
                action["id"]: target_answer["probabilities"][index]
                for index, action in targets[operation].items()
            }
        else:
            target_answer = None
            selected = controls[operation]["id"] if operation in controls else operation
            probabilities = {selected: operation_answer["probabilities"][operation]}
    except (KeyError, TypeError, ValueError, AttributeError):
        name = "OpenRouter" if provider == "openrouter" else provider.title()
        raise ValueError(f"Invalid {name} decision; no action executed.") from None
    return {
        "provider": provider, "model": result.get("model", config["model"]),
        "choice": selected, "operation": operation,
        "target": target_answer["choice"] if target_answer else None,
        "confidence": operation_answer["confidence"], "target_confidence": (
            target_answer["confidence"] if target_answer else None
        ),
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "usage": result.get("usage", {}), "raw_answers": result["answers"], "request": request,
    }


def choose(state, goal, history):
    config = decision_config()
    page_text = state["text"][:12000]
    elements, targets, controls = action_space(state["actions"], max_actions=80)
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded", "autocomplete_value") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": config["model"],
        "state": {
            "page": {"url": state["url"], "title": state["title"], "text": page_text},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed", "url")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    key = os.environ.get(config["key_name"], "").strip()
    if not key:
        raise ValueError(f"Choosing an action needs {config['key_name']}; no model request or browser action was made.")
    if config["provider"] == "openrouter":
        return choose_openrouter(body, targets, controls, config, key)
    started = time.perf_counter()
    result = post_typesafe_json("https://api.typesafe.ai/v1/systemone", key, body, config)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }



def validate_browser_url(url):
    if not isinstance(url, str) or not url or any(char.isspace() or ord(char) < 32 for char in url):
        raise ValueError("Start URL must be an absolute HTTP(S) URL without whitespace or control characters.")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        raise ValueError("Start URL has an invalid hostname or port.") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Start URL must be an absolute HTTP(S) URL without embedded credentials.")
    return url


def text_json(system, context, *, purpose, max_tokens=2048):
    key = os.environ.get("TEXT_MODEL_API_KEY", "").strip()
    if not key:
        raise ValueError(f"{purpose} needs TEXT_MODEL_API_KEY; no model request was made.")
    base = text_base_url()
    model = os.environ.get("TEXT_MODEL", "deepseek-chat").strip()
    if not model:
        raise ValueError("TEXT_MODEL cannot be blank.")
    endpoint = endpoint_host(base)
    reasoning = ({"thinking": {"type": "disabled"}} if endpoint == "api.deepseek.com"
                 else {"reasoning": {"effort": "low"}})
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions", key,
        {"model": model, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
         **reasoning, "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]},
        provider="text helper", model=model,
    )
    reason = "malformed completion response"
    try:
        completion = result["choices"][0]
        message = completion["message"]
        content = message.get("content")
        if message.get("refusal"):
            reason = "provider refused the request"
            raise ValueError()
        if completion.get("finish_reason") == "length":
            reason = "output truncated by the token limit"
            raise ValueError()
        if not isinstance(content, str) or not content.strip():
            reason = "empty or non-text completion"
            raise ValueError()
        content = content.strip()
        lines = content.splitlines()
        if len(lines) >= 3 and lines[0].lower() in {"```json", "```"} and lines[-1] == "```":
            content = "\n".join(lines[1:-1])
        reason = "completion is not a JSON object"
        output = json.loads(content)
        if not isinstance(output, dict):
            raise ValueError()
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise ValueError(
            f"Text helper returned invalid JSON for {purpose}: {reason} "
            f"[{endpoint} · {model}]."
        ) from None
    return output, {"model": model, "latency_ms": round((time.perf_counter() - started) * 1000),
                    "usage": result.get("usage", {}), "purpose": purpose}


def choose_start_url(goal):
    output, helper = text_json(START_URL, {"goal": goal}, purpose="automatic site selection", max_tokens=512)
    if set(output) != {"url"}:
        raise ValueError("Text helper returned no valid start URL; browser was not opened.")
    try:
        url = validate_browser_url(output["url"])
    except ValueError:
        raise ValueError("Text helper returned no valid start URL; browser was not opened.") from None
    return url, helper


def final_report(goal, evidence, stop_reason, truncated):
    allowed = {item["url"] for item in evidence}
    output, helper = text_json(
        FINAL_REPORT, {"goal": goal, "stop_reason": stop_reason, "evidence_truncated": truncated,
                       "allowed_sources": sorted(allowed), "evidence": evidence},
        purpose="final report", max_tokens=4096,
    )
    try:
        answer, sources, limitations = output["answer"], output["sources"], output["limitations"]
        valid = (set(output) == {"answer", "sources", "limitations"} and isinstance(answer, str)
                 and answer.strip() and isinstance(sources, list)
                 and all(isinstance(x, str) and x in allowed for x in sources)
                 and isinstance(limitations, list) and all(isinstance(x, str) for x in limitations))
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Text helper returned an invalid or ungrounded final report; evidence was preserved.")
    return {"answer": answer, "sources": sources, "limitations": limitations}, helper

def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    try:
        output, helper = text_json(TEXT_VALUE, context, purpose="TYPE_TEXT", max_tokens=1024)
    except ValueError as error:
        if "invalid JSON" in str(error):
            raise ValueError("Text helper returned no valid field value; nothing typed.") from None
        raise
    value = output.get("text")
    if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError("Text helper returned no valid field value; nothing typed.")
    return value, helper
