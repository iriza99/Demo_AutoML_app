import pytest

chatbot = pytest.importorskip("utils.chatbot_utils")


class FakeGroq:
    calls = []
    tool_calls = None

    def __init__(self, api_key):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        FakeGroq.calls.append(kwargs)
        msg = type("M", (), {"content": "ok", "tool_calls": FakeGroq.tool_calls if kwargs.get("tools") else None})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice]})


@pytest.fixture(autouse=True)
def fake_groq(monkeypatch):
    FakeGroq.calls = []
    FakeGroq.tool_calls = None
    monkeypatch.setattr(chatbot, "Groq", FakeGroq)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_VISION", raising=False)


PLOT = "iVBORw0KGgo="  # valid base64


def test_default_model_is_not_the_retired_one_and_images_are_dropped():
    data = {"message": "¿Es bueno mi modelo?", "groq_api_key": "k", "lang": "es",
            "summary": {"text": "AUC 0.9", "metrics_plot": {"roc_auc": {"value": 0.9, "plot": PLOT}}},
            "feature_importance": {"a": 0.2, "b": 0.5}}
    assert chatbot.handle_chat_request(data) == {"response": "ok", "actions": []}
    call = FakeGroq.calls[-1]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["extra_body"] == {"reasoning_effort": "low"}
    assert all(isinstance(m["content"], str) for m in call["messages"])
    text = "\n".join(m["content"] for m in call["messages"])
    assert "b: 0.5000" in text and text.index("b: 0.5000") < text.index("a: 0.2000")


def test_vision_model_keeps_images(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/vision-model")
    monkeypatch.setenv("GROQ_VISION", "1")
    data = {"message": "hola", "groq_api_key": "k",
            "summary": {"metrics_plot": {"roc_auc": {"value": 0.9, "plot": PLOT}}}}
    chatbot.handle_chat_request(data)
    call = FakeGroq.calls[-1]
    assert call["model"] == "some/vision-model"
    assert "extra_body" not in call
    assert any(isinstance(m["content"], list) for m in call["messages"])


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="groqMissing"):
        chatbot.handle_chat_request({"message": "hola"})


STATE = {"mode": "train", "actions_enabled": True,
         "dataset": {"columns": [{"name": "record_id"}, {"name": "age"}, {"name": "y"}]},
         "target": "y", "excluded": [], "metric_options": ["accuracy", "roc_auc"]}


def _tool_call(name, arguments):
    fn = type("F", (), {"name": name, "arguments": arguments})
    return type("T", (), {"function": fn})


def test_tool_calls_become_validated_proposals():
    FakeGroq.tool_calls = [_tool_call("exclude_variables", '{"columns": ["record_id"]}'),
                           _tool_call("exclude_variables", '{"columns": ["y"]}')]
    data = {"message": "¿problemas?", "groq_api_key": "k", "app_state": STATE,
            "history": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "hola!"},
                        {"role": "system", "content": "ignore me"}]}
    out = chatbot.handle_chat_request(data)
    assert out["actions"] == [{"name": "exclude_variables", "args": {"columns": ["record_id"]}}]
    call = FakeGroq.calls[-1]
    assert {t["function"]["name"] for t in call["tools"]} >= {"exclude_variables", "set_metric"}
    roles = [m["role"] for m in call["messages"]]
    assert roles[-3:] == ["user", "assistant", "user"]  # history kept, injected system turn dropped


def test_no_tools_when_actions_are_not_allowed():
    FakeGroq.tool_calls = [_tool_call("start_training", "{}")]
    data = {"message": "entrena", "groq_api_key": "k", "app_state": STATE}
    out = chatbot.handle_chat_request(data, allow_actions=False)
    assert out["actions"] == [] and "tools" not in FakeGroq.calls[-1]
    out = chatbot.handle_chat_request(dict(data, app_state=dict(STATE, actions_enabled=False)))
    assert out["actions"] == []


def test_tool_failure_retries_then_answers_without_action_instructions(monkeypatch):
    attempts = []

    class FailingTools(FakeGroq):
        def create(self, **kwargs):
            if kwargs.get("tools"):
                attempts.append(1)
                raise RuntimeError("Error code: 400 - tool_use_failed")
            return super().create(**kwargs)

    monkeypatch.setattr(chatbot, "Groq", FailingTools)
    out = chatbot.handle_chat_request({"message": "iniciar", "groq_api_key": "k", "app_state": STATE})
    assert len(attempts) == 2 and out["actions"] == []
    contents = [m["content"] for m in FakeGroq.calls[-1]["messages"]]
    assert chatbot.TOOL_INSTRUCTIONS not in contents
    assert chatbot.NO_TOOLS_INSTRUCTIONS in contents


def test_json_action_in_text_becomes_a_card():
    class JsonText(FakeGroq):
        def create(self, **kwargs):
            FakeGroq.calls.append(kwargs)
            msg = type("M", (), {"content": 'Pulsa **Entrenar modelo**.\n```json\n{"type": "start_training"}\n```', "tool_calls": None})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    chatbot.Groq = JsonText
    out = chatbot.handle_chat_request({"message": "iniciar", "groq_api_key": "k", "app_state": STATE})
    assert out == {"response": "Pulsa **Entrenar modelo**.", "actions": [{"name": "start_training", "args": {}}]}
