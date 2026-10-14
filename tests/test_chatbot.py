import pytest

chatbot = pytest.importorskip("utils.chatbot_utils")


class FakeGroq:
    calls = []

    def __init__(self, api_key):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        FakeGroq.calls.append(kwargs)
        msg = type("M", (), {"content": "ok"})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice]})


@pytest.fixture(autouse=True)
def fake_groq(monkeypatch):
    FakeGroq.calls = []
    monkeypatch.setattr(chatbot, "Groq", FakeGroq)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_VISION", raising=False)


PLOT = "iVBORw0KGgo="  # valid base64


def test_default_model_is_not_the_retired_one_and_images_are_dropped():
    data = {"message": "¿Es bueno mi modelo?", "groq_api_key": "k", "lang": "es",
            "summary": {"text": "AUC 0.9", "metrics_plot": {"roc_auc": {"value": 0.9, "plot": PLOT}}},
            "feature_importance": {"a": 0.2, "b": 0.5}}
    assert chatbot.handle_chat_request(data) == "ok"
    call = FakeGroq.calls[-1]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["reasoning_effort"] == "low"
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
    assert "reasoning_effort" not in call
    assert any(isinstance(m["content"], list) for m in call["messages"])


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="groqMissing"):
        chatbot.handle_chat_request({"message": "hola"})
