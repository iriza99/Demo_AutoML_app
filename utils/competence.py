"""Implicit two-dimensional competence model (ML x clinical domain).

The model never asks the user for their level. It updates a posterior over
three levels (1 = low, 2 = medium, 3 = high) for each dimension from
interaction telemetry, using hand-specified likelihoods (a transparent
baseline that can be reported and audited). When a typed decision layer is
configured, a second estimate is obtained from it over a structured summary
of the same telemetry; both estimates are logged so they can be compared
against the objective tests in the study.

Behaviour that happened *after* the system pointed at an issue is not taken
as evidence of competence for that issue.
"""
import re
import time

import numpy as np

from utils.typed_decision import confidence_from_probs, safe_call

DIMENSIONS = ("ml", "domain")
LEVELS = (1, 2, 3)

# P(evidence | level) for levels (low, medium, high)
STRONG_HIGH = (0.10, 0.40, 0.90)
MODERATE_HIGH = (0.30, 0.60, 0.90)
WEAK_HIGH = (0.50, 0.70, 0.90)
WEAK_LOW = (0.90, 0.70, 0.50)
MODERATE_LOW = (0.90, 0.60, 0.30)
STRONG_LOW = (0.90, 0.40, 0.10)

ML_TERMS = re.compile(
    r"\b(auc|roc|f1|recall|precision|precisi[oó]n|sensibilidad|especificidad|overfit\w*|"
    r"sobreajust\w*|leak\w*|fuga\w*|cross[\s-]?validation|validaci[oó]n cruzada|shap|"
    r"hiperpar[aá]metr\w*|hyperparam\w*|xgboost|lightgbm|catboost|gradient boosting|"
    r"random forest|desbalance\w*|imbalanc\w*|calibra\w*|baseline|holdout|bagging|ensemble|"
    r"feature importance|regulari[sz]\w*|oof|out[\s-]of[\s-]fold)\b",
    re.IGNORECASE,
)
DOMAIN_TERMS = re.compile(
    r"\b(nyha|fevi|lvef|nt[\s-]?probnp|bnp|sepsis[\s-]?3|q?sofa|lactato|lactate|"
    r"[uµ]mol|mg/dl|mmol|triaje|triage|diur[eé]tic\w*|furosemid\w*|betabloque\w*|"
    r"beta[\s-]?blocker|ieca|arni|antibi[oó]tic\w*|creatinin\w*|insuficiencia card\w*|"
    r"heart failure|fibrilaci[oó]n|glasgow|comorbilidad\w*|reingreso|readmission)\b",
    re.IGNORECASE,
)
BASIC_QUESTION = re.compile(
    r"(qu[eé] (es|son|significa|quiere decir)|what (is|are|does)|c[oó]mo se interpreta|"
    r"no entiendo|i don'?t understand|explain|expl[ií]ca(me)?)",
    re.IGNORECASE,
)
ML_SETTINGS_EVENTS = {"cv_toggle", "val_size_change", "model_toggle"}
BALANCED_METRICS = {"roc_auc", "f1_macro", "f1", "balanced_accuracy", "average_precision"}


class CompetenceModel:
    def __init__(self, prior=None):
        prior = np.array(prior or (1 / 3, 1 / 3, 1 / 3), dtype=float)
        self.log_post = {d: np.log(prior) for d in DIMENSIONS}
        self.trace = []
        self.counts = {}
        self.chat_messages = []
        self.step_times = {}
        self._step_started = None
        self._current_step = None
        self.started = time.time()

    # ── Evidence bookkeeping ────────────────────────────────────────────────
    def _update(self, dim, likelihood, reason, ts=None):
        self.log_post[dim] = self.log_post[dim] + np.log(np.array(likelihood))
        self.log_post[dim] -= np.logaddexp.reduce(self.log_post[dim])
        self.trace.append({"ts": ts, "dimension": dim, "reason": reason,
                           "likelihood": list(likelihood)})

    def estimate(self, dim):
        probs = np.exp(self.log_post[dim])
        probs = probs / probs.sum()
        return {
            "probs": [round(float(p), 4) for p in probs],
            "score": round(float(np.dot(probs, LEVELS)), 3),
            "confidence": round(confidence_from_probs(list(probs)), 3),
            "source": "heuristic",
        }

    def estimates(self):
        return {d: self.estimate(d) for d in DIMENSIONS}

    # ── Event interpretation ────────────────────────────────────────────────
    def observe(self, event, context):
        """Update from one telemetry event.

        context provides dataset knowledge without revealing the answer key:
        {"identifiers": set, "leak_candidates": set, "unit_mixed": set,
         "minority_share": float|None, "prompted": set of issue keys/columns
         already pointed out by the system}
        """
        etype = event.get("type")
        ts = event.get("ts")
        self.counts[etype] = self.counts.get(etype, 0) + 1
        prompted = context.get("prompted", set())

        if etype == "step_view":
            self._track_step(event.get("step"), ts)
        elif etype == "metric_change":
            metric = str(event.get("metric", "")).lower()
            imbalanced = (context.get("minority_share") or 1) < 0.2
            if metric in BALANCED_METRICS and "imbalance_metric" not in prompted:
                self._update("ml", STRONG_HIGH if imbalanced else MODERATE_HIGH,
                             f"chose {metric}", ts)
        elif etype == "advanced_toggle" and event.get("open"):
            if self.counts[etype] == 1:
                self._update("ml", WEAK_HIGH, "opened advanced options", ts)
        elif etype in ML_SETTINGS_EVENTS:
            if self.counts[etype] == 1:
                self._update("ml", MODERATE_HIGH, f"changed {etype}", ts)
        elif etype == "column_drop_toggle" and event.get("dropped"):
            col = event.get("column")
            if col in prompted:
                return
            if col in context.get("identifiers", set()):
                self._update("ml", MODERATE_HIGH, f"dropped identifier {col}", ts)
            elif col in context.get("leak_candidates", set()):
                self._update("domain", STRONG_HIGH, f"dropped late-information column {col}", ts)
            elif col in context.get("unit_mixed", set()):
                self._update("domain", MODERATE_HIGH, f"dropped unit-mixed column {col}", ts)
        elif etype == "shap_request":
            if self.counts[etype] == 1:
                self._update("ml", WEAK_HIGH, "requested SHAP", ts)
        elif etype == "chat_send":
            self._observe_chat(str(event.get("text", "")), ts)
        elif etype == "guidance_answer":
            dim = event.get("dimension")
            if dim in DIMENSIONS:
                if event.get("correct") is True:
                    self._update(dim, STRONG_HIGH, f"answered probe {event.get('key')} correctly", ts)
                elif event.get("answer") == "dont_know":
                    self._update(dim, MODERATE_LOW, f"did not know probe {event.get('key')}", ts)
                else:
                    self._update(dim, STRONG_LOW, f"answered probe {event.get('key')} wrongly", ts)
        elif etype == "train_start":
            self._observe_training_config(event, context, ts)

    def _observe_chat(self, text, ts):
        self.chat_messages.append(text[:300])
        is_basic = bool(BASIC_QUESTION.search(text))
        ml_hits = len(ML_TERMS.findall(text))
        domain_hits = len(DOMAIN_TERMS.findall(text))
        if ml_hits:
            self._update("ml", MODERATE_LOW if is_basic else MODERATE_HIGH,
                         "ML vocabulary in a basic question" if is_basic else "used ML vocabulary", ts)
        if domain_hits:
            self._update("domain", MODERATE_LOW if is_basic else MODERATE_HIGH,
                         "clinical vocabulary in a basic question" if is_basic else "used clinical vocabulary", ts)

    def _observe_training_config(self, event, context, ts):
        cfg = event.get("config", {})
        dropped = set(cfg.get("dropped") or [])
        metric = str(cfg.get("metric") or "accuracy").lower()
        prompted = context.get("prompted", set())
        if (context.get("minority_share") or 1) < 0.2 and metric == "accuracy" \
                and "imbalance_metric" not in prompted:
            self._update("ml", WEAK_LOW, "trained rare outcome with accuracy", ts)
        kept_ids = context.get("identifiers", set()) - dropped - prompted
        if kept_ids:
            self._update("ml", WEAK_LOW, "trained with identifier column", ts)
        kept_leaks = context.get("leak_candidates", set()) - dropped - prompted
        if kept_leaks:
            self._update("domain", WEAK_LOW, "trained with late-information column", ts)

    def _track_step(self, step, ts):
        now = ts if ts is not None else time.time() * 1000
        if self._current_step is not None and self._step_started is not None:
            spent = (now - self._step_started) / 1000.0
            self.step_times[self._current_step] = self.step_times.get(self._current_step, 0) + spent
        self._current_step, self._step_started = step, now

    # ── Decision-layer estimate ─────────────────────────────────────────────
    def summary(self):
        """Structured, model-agnostic description of the session so far."""
        return {
            "minutes_in_session": round((time.time() - self.started) / 60, 1),
            "event_counts": dict(self.counts),
            "seconds_per_wizard_step": {str(k): round(v) for k, v in self.step_times.items()},
            "self_initiated_actions": [t["reason"] for t in self.trace
                                       if t["likelihood"][2] > t["likelihood"][0]],
            "signs_of_difficulty": [t["reason"] for t in self.trace
                                    if t["likelihood"][0] > t["likelihood"][2]],
            "recent_chat_messages": self.chat_messages[-5:],
        }

    def decision_estimates(self, decision_layer):
        if decision_layer is None:
            return None
        questions = {
            "ml": ("How much machine-learning expertise does this user show while "
                   "building a predictive model?"),
            "domain": ("How much clinical/medical domain expertise does this user show "
                       "about the variables in the dataset?"),
        }
        levels = ["low", "medium", "high"]
        out = {}
        state = self.summary()
        for dim, question in questions.items():
            decision = safe_call(decision_layer.score, state, question, levels)
            if decision is None:
                return None
            est = decision.to_dict()
            out[dim] = {"probs": est["probs"], "score": est["expected"],
                        "confidence": est["confidence"], "source": "decision_layer"}
        return out
