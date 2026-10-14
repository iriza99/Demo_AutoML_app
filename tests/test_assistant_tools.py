from utils.assistant_tools import build_tools, parse_tool_calls, validate_call

STATE = {
    "mode": "train", "actions_enabled": True,
    "dataset": {"columns": [{"name": c} for c in ("record_id", "age", "nt_probnp_followup", "readmission_30d")]},
    "target": "readmission_30d", "excluded": ["age"],
    "metric_options": ["accuracy", "roc_auc", "f1_macro"], "training_running": False,
}


def names(tools):
    return {t["function"]["name"] for t in tools}


def call(name, args):
    import json
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def test_tools_depend_on_state():
    assert names(build_tools({})) == {"switch_mode"}
    tools = build_tools(STATE)
    assert {"exclude_variables", "include_variables", "set_metric", "start_training"} <= names(tools)
    assert "show_results_tab" not in names(tools)  # no results yet
    exclude = next(t for t in tools if t["function"]["name"] == "exclude_variables")
    assert exclude["function"]["parameters"]["properties"]["columns"]["items"]["enum"] == ["record_id", "nt_probnp_followup"]
    running = dict(STATE, training_running=True)
    assert names(build_tools(running)) == {"switch_mode", "go_to_step"}
    with_results = dict(STATE, results={"task_type": "binary", "tabs": ["performance", "shap"]})
    assert {"show_results_tab", "generate_shap"} <= names(build_tools(with_results))


def test_arguments_are_validated_against_the_state():
    assert validate_call("exclude_variables", {"columns": ["record_id", "made_up", "readmission_30d"]}, STATE) == {"columns": ["record_id"]}
    assert validate_call("exclude_variables", {"columns": ["readmission_30d"]}, STATE) is None  # target
    assert validate_call("include_variables", {"columns": ["age"]}, STATE) == {"columns": ["age"]}
    assert validate_call("set_metric", {"metric": "r2"}, STATE) is None
    assert validate_call("set_training_time", {"minutes": 999}, STATE) == {"minutes": 180}
    assert validate_call("set_target", {"column": "nope"}, STATE) is None
    assert validate_call("go_to_step", {"step": "results"}, STATE) is None


def test_parse_drops_unknown_invalid_and_duplicate_calls():
    calls = [
        call("exclude_variables", {"columns": ["record_id"]}),
        call("exclude_variables", {"columns": ["record_id"]}),
        call("delete_everything", {}),
        {"function": {"name": "set_metric", "arguments": "{not json"}},
        call("set_metric", {"metric": "roc_auc"}),
    ]
    assert parse_tool_calls(calls, STATE) == [
        {"name": "exclude_variables", "args": {"columns": ["record_id"]}},
        {"name": "set_metric", "args": {"metric": "roc_auc"}},
    ]
