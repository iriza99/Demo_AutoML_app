"""Actions the assistant can propose in the app (LLM tool calling).

The model never changes anything itself. It returns tool calls, which the
server validates against the current app state and sends to the browser as
proposals; the user applies each one with a click. Only tools that make sense
in the current state are offered, and arguments are restricted to values that
exist (real column names, metrics available for the task, ...).
"""
import json
import logging
import re

MAX_ENUM = 300  # above this many columns, names are validated but not enumerated

STEPS = ("data", "train", "results")
TABS = ("performance", "importance", "shap", "models")


def _fn(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def _columns_schema(names, description):
    items = {"type": "string"}
    if 0 < len(names) <= MAX_ENUM:
        items["enum"] = list(names)
    return {"type": "array", "items": items, "minItems": 1, "description": description}


def sanitize_state(state):
    """Keep only the fields the tools need, with safe types and sizes."""
    if not isinstance(state, dict):
        return {}
    dataset = state.get("dataset") if isinstance(state.get("dataset"), dict) else None
    columns = []
    if dataset:
        for col in (dataset.get("columns") or [])[:2000]:
            if isinstance(col, dict) and isinstance(col.get("name"), str):
                columns.append(col["name"])
    results = state.get("results") if isinstance(state.get("results"), dict) else None
    return {
        "mode": state.get("mode") if state.get("mode") in ("train", "predict") else "train",
        "columns": columns,
        "target": state.get("target") if state.get("target") in columns else None,
        "excluded": [c for c in (state.get("excluded") or []) if c in columns],
        "metric_options": [m for m in (state.get("metric_options") or []) if isinstance(m, str)][:20],
        "training_running": bool(state.get("training_running")),
        "has_results": bool(results),
        "task_type": (results or {}).get("task_type"),
        "result_tabs": [t for t in ((results or {}).get("tabs") or []) if t in TABS],
    }


def build_tools(state):
    """Tool definitions valid for the current app state."""
    s = sanitize_state(state)
    tools = [_fn("switch_mode", "Switch between training a model and predicting with a saved model.",
                 {"mode": {"type": "string", "enum": ["train", "predict"]}}, ["mode"])]
    if not s["columns"]:
        return tools

    candidates = [c for c in s["columns"] if c != s["target"]]
    included = [c for c in candidates if c not in s["excluded"]]
    steps = list(STEPS) if s["has_results"] else ["data", "train"]
    tools.append(_fn("go_to_step", "Show one of the app steps.",
                     {"step": {"type": "string", "enum": steps}}, ["step"]))
    if s["training_running"]:
        return tools

    if included:
        tools.append(_fn(
            "exclude_variables",
            "Stop the model from using these variables (e.g. identifiers, information recorded after "
            "the prediction moment, or columns that leak the outcome). Reversible.",
            {"columns": _columns_schema(included, "Exact column names to exclude.")}, ["columns"]))
    if s["excluded"]:
        tools.append(_fn("include_variables", "Let the model use previously excluded variables again.",
                         {"columns": _columns_schema(s["excluded"], "Exact column names to include.")},
                         ["columns"]))
    tools.append(_fn("set_target", "Choose the column the model must predict.",
                     {"column": {"type": "string", **({"enum": s["columns"]} if len(s["columns"]) <= MAX_ENUM else {})}},
                     ["column"]))
    if s["metric_options"]:
        tools.append(_fn("set_metric", "Choose the metric AutoML optimises and reports first.",
                         {"metric": {"type": "string", "enum": s["metric_options"]}}, ["metric"]))
    tools.append(_fn("set_training_time", "Set the maximum training time in minutes.",
                     {"minutes": {"type": "integer", "minimum": 1, "maximum": 180}}, ["minutes"]))
    tools.append(_fn("set_cross_validation",
                     "Turn cross-validation on (recommended) or off (a holdout split is used instead).",
                     {"enabled": {"type": "boolean"}}, ["enabled"]))
    tools.append(_fn("start_training", "Start training with the current configuration."))
    if s["has_results"]:
        tabs = s["result_tabs"] or ["performance"]
        tools.append(_fn("show_results_tab", "Open a tab of the results.",
                         {"tab": {"type": "string", "enum": tabs}}, ["tab"]))
        if "shap" in tabs:
            tools.append(_fn("generate_shap", "Compute the SHAP plot that explains individual predictions."))
    return tools


def validate_call(name, args, state):
    """Return cleaned arguments, or None when the call does not fit the state."""
    s = sanitize_state(state)
    columns = set(s["columns"])
    if not isinstance(args, dict):
        return None
    if name == "switch_mode":
        return {"mode": args["mode"]} if args.get("mode") in ("train", "predict") else None
    if not columns:
        return None
    if name == "go_to_step":
        allowed = STEPS if s["has_results"] else ("data", "train")
        return {"step": args["step"]} if args.get("step") in allowed else None
    if s["training_running"]:
        return None
    if name in ("exclude_variables", "include_variables"):
        cols = args.get("columns")
        if not isinstance(cols, list):
            return None
        pool = (set(s["excluded"]) if name == "include_variables"
                else columns - set(s["excluded"]) - {s["target"]})
        cols = [c for c in dict.fromkeys(cols) if isinstance(c, str) and c in pool]
        return {"columns": cols} if cols else None
    if name == "set_target":
        return {"column": args["column"]} if args.get("column") in columns else None
    if name == "set_metric":
        return {"metric": args["metric"]} if args.get("metric") in s["metric_options"] else None
    if name == "set_training_time":
        try:
            minutes = int(args.get("minutes"))
        except (TypeError, ValueError):
            return None
        return {"minutes": max(1, min(180, minutes))}
    if name == "set_cross_validation":
        return {"enabled": args["enabled"]} if isinstance(args.get("enabled"), bool) else None
    if name == "start_training":
        return {}
    if name == "show_results_tab" and s["has_results"]:
        return {"tab": args["tab"]} if args.get("tab") in (s["result_tabs"] or ["performance"]) else None
    if name == "generate_shap" and s["has_results"] and "shap" in s["result_tabs"]:
        return {}
    return None


def parse_tool_calls(tool_calls, state):
    """Turn the model's tool calls into validated, de-duplicated action proposals."""
    actions = []
    seen = set()
    offered = {t["function"]["name"] for t in build_tools(state)}
    for call in tool_calls or []:
        fn = getattr(call, "function", None) or (call.get("function") if isinstance(call, dict) else None)
        if fn is None:
            continue
        name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
        raw = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
        if name not in offered:
            continue
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except json.JSONDecodeError:
            logging.warning(f"Assistant proposed {name} with invalid JSON arguments")
            continue
        clean = validate_call(name, args, state)
        if clean is None:
            continue
        key = (name, json.dumps(clean, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        actions.append({"name": name, "args": clean})
    return actions[:6]


_NAME_KEYS = ("name", "type", "action", "tool", "function")
_ARG_KEYS = ("args", "arguments", "parameters", "params")
_FENCED = re.compile(r"```[a-zA-Z]*\s*(\{.*?\})\s*```", re.DOTALL)


def _objects_in(text):
    """JSON objects written in the text: fenced blocks first, then bare {...} spans."""
    found = [(m.start(), m.end(), m.group(1)) for m in _FENCED.finditer(text)]
    taken = [(a, b) for a, b, _ in found]
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and not any(a <= start < b for a, b in taken):
                found.append((start, i + 1, text[start:i + 1]))
    return found


def extract_text_actions(text, state):
    """Rescue actions the model wrote as JSON in its answer instead of calling a tool.

    Returns (text without those JSON blocks, validated actions).
    """
    if not text or "{" not in text:
        return text, []
    offered = {t["function"]["name"] for t in build_tools(state)}
    actions, spans = [], []
    for start, end, raw in _objects_in(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str) and obj[k] in offered), None)
        if name is None:
            continue
        args = next((obj[k] for k in _ARG_KEYS if isinstance(obj.get(k), dict)), None)
        if args is None:
            args = {k: v for k, v in obj.items() if k not in _NAME_KEYS}
        clean = validate_call(name, args, state)
        if clean is not None:
            actions.append({"name": name, "args": clean})
        spans.append((start, end))
    if not spans:
        return text, []
    # Cut the JSON (and a short heading right before it, e.g. "**Proposed action**")
    out, last = [], 0
    for start, end in sorted(spans):
        chunk = text[last:start]
        lines = chunk.rstrip().split("\n")
        if len(lines) > 1 and len(lines[-1]) < 60 and re.match(r"^\s*(#+|\*\*|__)", lines[-1]):
            chunk = "\n".join(lines[:-1])
        out.append(chunk)
        last = end
    out.append(text[last:])
    cleaned = re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
    unique = []
    for a in actions:
        if a not in unique:
            unique.append(a)
    return cleaned, unique[:6]


TOOL_INSTRUCTIONS = """
You can PROPOSE actions in the app by calling the provided tools. Rules:
- Calling a tool does not change anything: the user sees each proposal as a card and decides whether to apply it. Never say that you already did something.
- Only propose actions that follow from the user's request or that clearly help with it (for example, excluding an identifier or a variable recorded after the prediction moment when you point out that problem). Explain why in your answer.
- Do not propose start_training unless the user asks to train or agrees to it.
- Use the exact column names from the app state. If you are unsure, ask instead of proposing.
- Always write a short answer in the user's language alongside any proposal.
- Never write actions as JSON or code in your answer: use the tools.
- When you mention buttons or sections, use the labels in ui_labels of the app state.
""".strip()

NO_TOOLS_INSTRUCTIONS = """
Actions cannot be proposed right now. Tell the user in words which button or option to use,
with the labels in ui_labels of the app state. Never write JSON or code describing actions.
""".strip()
