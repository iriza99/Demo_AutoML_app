"""Actions the assistant can propose in the app (LLM tool calling).

The model never changes anything itself. It returns tool calls, which the
server validates against the current app state and sends to the browser as
proposals; the user applies each one with a click. Only tools that make sense
in the current state are offered, and arguments are restricted to values that
exist (real column names, metrics available for the task, ...).
"""
import json
import logging

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


TOOL_INSTRUCTIONS = """
You can PROPOSE actions in the app by calling the provided tools. Rules:
- Calling a tool does not change anything: the user sees each proposal as a card and decides whether to apply it. Never say that you already did something.
- Only propose actions that follow from the user's request or that clearly help with it (for example, excluding an identifier or a variable recorded after the prediction moment when you point out that problem). Explain why in your answer.
- Do not propose start_training unless the user asks to train or agrees to it.
- Use the exact column names from the app state. If you are unsure, ask instead of proposing.
- Always write a short answer in the user's language alongside any proposal.
""".strip()
