"""Jev-style typed decision layer running on a local open-weights model.

The layer answers three kinds of typed questions about a structured state,
mirroring the Choice / Score / Noul question types popularised by TypeSafe
AI's Jev, but with a local backend so that no study data leaves the machine:

* choice(state, question, options) -> Decision over the options
* score(state, question, levels)    -> Decision over ordered levels, with an
                                       expected value in [1, len(levels)]
* noul(state, question)             -> probability of "yes"

Answers are read from the next-token log-probabilities of the option letters
("A", "B", ...) in a single forward pass, then renormalised over the allowed
letters (masked softmax). No text is generated.

The backend is any OpenAI-compatible server that returns `top_logprobs`
(llama.cpp server, vLLM, recent Ollama versions). Configure it with:

    DECISION_BASE_URL=http://localhost:8080      # server root, without /v1
    DECISION_MODEL=qwen2.5-7b-instruct
    DECISION_TIMEOUT=8                            # seconds

If DECISION_BASE_URL is unset the layer is disabled and callers fall back to
their heuristic estimates.
"""
import json
import logging
import math
import os
import string
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a decision engine. Read the state and the question, then answer "
    "with the single capital letter of the best option. Do not explain."
)
# Log-probability assigned to an option letter missing from top_logprobs
MISSING_LOGPROB_MARGIN = 5.0


@dataclass
class Decision:
    options: list
    probs: list
    source: str = "decision_layer"
    raw: dict = field(default_factory=dict)

    @property
    def best(self):
        return self.options[max(range(len(self.probs)), key=self.probs.__getitem__)]

    @property
    def expected(self):
        """Probability-weighted level (1-based), as in a Jev Score answer."""
        return sum((i + 1) * p for i, p in enumerate(self.probs))

    @property
    def confidence(self):
        return confidence_from_probs(self.probs)

    def to_dict(self):
        return {
            "options": self.options,
            "probs": [round(p, 4) for p in self.probs],
            "expected": round(self.expected, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def confidence_from_probs(probs):
    """1 - normalised entropy: 1 for a one-hot distribution, 0 for uniform."""
    k = len(probs)
    if k < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - entropy / math.log(k))


def masked_softmax(top_logprobs, n_options):
    """Turn top-k next-token log-probabilities into a distribution over letters.

    `top_logprobs` is a list of {"token": str, "logprob": float}. Tokens are
    matched after stripping whitespace and punctuation, so " A", "A)" and
    "A" all count for option A.
    """
    letters = string.ascii_uppercase[:n_options]
    best = {}
    for item in top_logprobs:
        token = str(item.get("token", "")).strip().strip(".):").upper()
        if token in letters:
            best[token] = max(best.get(token, -math.inf), float(item["logprob"]))
    if not best:
        raise ValueError("no option letter in top_logprobs")
    floor = min(best.values()) - MISSING_LOGPROB_MARGIN
    logits = [best.get(letter, floor) for letter in letters]
    top = max(logits)
    exp = [math.exp(v - top) for v in logits]
    total = sum(exp)
    return [v / total for v in exp]


def build_prompt(state, question, options):
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False, indent=1)
    lines = [f"STATE:\n{state}", f"\nQUESTION: {question}", "\nOPTIONS:"]
    for letter, option in zip(string.ascii_uppercase, options):
        lines.append(f"{letter}) {option}")
    lines.append("\nAnswer with one letter.")
    return "\n".join(lines)


class LocalLogprobBackend:
    """OpenAI-compatible chat completions endpoint that exposes logprobs."""

    def __init__(self, base_url, model, timeout=8.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout

    def next_token_logprobs(self, prompt):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["logprobs"]["content"]
        return content[0]["top_logprobs"]


class TypedDecisionLayer:
    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def from_env(cls):
        base_url = os.getenv("DECISION_BASE_URL")
        if not base_url:
            return None
        model = os.getenv("DECISION_MODEL", "local-model")
        timeout = float(os.getenv("DECISION_TIMEOUT", "8"))
        return cls(LocalLogprobBackend(base_url, model, timeout))

    def _ask(self, state, question, options):
        prompt = build_prompt(state, question, options)
        top = self.backend.next_token_logprobs(prompt)
        return Decision(options=list(options), probs=masked_softmax(top, len(options)),
                        raw={"prompt": prompt})

    def choice(self, state, question, options):
        return self._ask(state, question, options)

    def score(self, state, question, levels):
        if not 2 <= len(levels) <= 10:
            raise ValueError("a score needs between 2 and 10 ordered levels")
        return self._ask(state, question, levels)

    def noul(self, state, question):
        decision = self._ask(state, question, ["yes", "no"])
        return decision.probs[0]


def safe_call(fn, *args, **kwargs):
    """Run a decision-layer call, returning None on any backend failure."""
    try:
        return fn(*args, **kwargs)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError,
            ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Decision layer unavailable: {type(exc).__name__}: {exc}")
        return None
