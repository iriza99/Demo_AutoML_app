import math
import urllib.error

import pytest

from utils.typed_decision import (Decision, TypedDecisionLayer, confidence_from_probs,
                                  masked_softmax, safe_call)


class FakeBackend:
    def __init__(self, top):
        self.top = top
        self.prompts = []

    def next_token_logprobs(self, prompt):
        self.prompts.append(prompt)
        return self.top


def test_masked_softmax_ignores_other_tokens_and_merges_variants():
    top = [{"token": " A", "logprob": math.log(0.2)}, {"token": "B)", "logprob": math.log(0.6)},
           {"token": "The", "logprob": math.log(0.1)}, {"token": "a", "logprob": math.log(0.05)}]
    probs = masked_softmax(top, 3)
    assert probs[1] > probs[0] > probs[2]
    assert abs(sum(probs) - 1) < 1e-9


def test_masked_softmax_without_any_option_raises():
    with pytest.raises(ValueError):
        masked_softmax([{"token": "Hello", "logprob": -0.1}], 2)


def test_score_returns_expected_level_and_confidence():
    top = [{"token": "A", "logprob": math.log(0.1)}, {"token": "B", "logprob": math.log(0.2)},
           {"token": "C", "logprob": math.log(0.7)}]
    layer = TypedDecisionLayer(FakeBackend(top))
    decision = layer.score({"x": 1}, "How expert?", ["low", "medium", "high"])
    assert decision.best == "high"
    assert 2.5 < decision.expected < 2.7
    assert 0 < decision.confidence < 1
    assert "C) high" in layer.backend.prompts[0]


def test_noul_returns_probability_of_yes():
    top = [{"token": "A", "logprob": math.log(0.9)}, {"token": "B", "logprob": math.log(0.1)}]
    assert abs(TypedDecisionLayer(FakeBackend(top)).noul("s", "q?") - 0.9) < 1e-9


def test_score_rejects_too_many_levels():
    with pytest.raises(ValueError):
        TypedDecisionLayer(FakeBackend([])).score("s", "q", [str(i) for i in range(11)])


def test_confidence_bounds():
    assert confidence_from_probs([1 / 3] * 3) == pytest.approx(0)
    assert confidence_from_probs([1, 0, 0]) == pytest.approx(1)
    assert Decision(["a", "b"], [0.5, 0.5]).confidence == pytest.approx(0)


def test_safe_call_swallows_backend_errors():
    def boom():
        raise urllib.error.URLError("down")
    assert safe_call(boom) is None


def test_from_env_disabled_without_url(monkeypatch):
    monkeypatch.delenv("DECISION_BASE_URL", raising=False)
    assert TypedDecisionLayer.from_env() is None
