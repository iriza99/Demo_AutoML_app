"""Study sessions: telemetry logging, competence tracking and guidance policy.

Conditions (set per session from the study URL):
    baseline  the published app: telemetry and competence are logged, but no
              guardrail interventions are shown
    generic   every detected issue is shown to everyone with the same
              intermediate-level text (blocking before training)
    adaptive  competence-aware: probe when unsure, otherwise tailor the
              explanation level and only block for low competence
"""
import json
import logging
import os
import re
import threading
import time
import uuid

from utils import guardrails
from utils.competence import CompetenceModel
from utils.study_messages import (format_facts, profile_instruction, render_probe,
                                  render_text, render_title)
from utils.typed_decision import TypedDecisionLayer

logger = logging.getLogger(__name__)

CONDITIONS = ("baseline", "generic", "adaptive")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "study", "datasets")
LOG_DIR = os.getenv("STUDY_LOG_DIR", os.path.join(BASE_DIR, "study_logs"))
DATASET_KEYS = {"A": "hf_readmission", "B": "ed_sepsis"}

# Policy thresholds on the 1-3 competence score
ASK_CONFIDENCE = 0.2
NOVICE_BELOW = 1.7
EXPERT_ABOVE = 2.4

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def load_dictionary(dataset_key):
    name = DATASET_KEYS.get(dataset_key)
    if not name:
        return None
    path = os.path.join(DATASET_DIR, f"{name}_dictionary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class StudySession:
    def __init__(self, participant_id, condition, dataset_key=None, lang="es",
                 decision_layer=None, competence_source="heuristic"):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        self.id = uuid.uuid4().hex[:12]
        self.participant_id = _SAFE_ID.sub("", participant_id or "anon")[:32] or "anon"
        self.condition = condition
        self.dataset_key = dataset_key
        self.lang = lang
        self.decision_layer = decision_layer
        self.competence_source = competence_source
        self.competence = CompetenceModel()
        self.dictionary = load_dictionary(dataset_key)
        self.profile = None
        self.df = None
        self.tinfo = None
        self.semantic = {}
        self.prompted = set()   # issue keys and columns already pointed out
        self.probed = set()     # issue keys already asked about
        self.shown = set()      # (issue key, mode, stage) already returned
        self.lock = threading.Lock()
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, f"{self.participant_id}_{self.id}.jsonl")
        self.log({"type": "session_created", "condition": condition,
                  "dataset_key": dataset_key, "lang": lang,
                  "competence_source": competence_source,
                  "decision_layer": decision_layer is not None})

    # ── Logging ─────────────────────────────────────────────────────────────
    def log(self, record):
        record = {"server_ts": round(time.time() * 1000), "session_id": self.id,
                  "participant_id": self.participant_id, **record}
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _context(self):
        profile = self.profile or {}
        leak_cols = set(self.semantic)
        if self.tinfo:
            leak_cols |= {c for c, auc in self.tinfo.get("single_feature_auc", {}).items()
                          if auc >= guardrails.LEAK_AUC_THRESHOLD
                          and c not in profile.get("identifiers", [])}
        return {
            "identifiers": set(profile.get("identifiers", [])),
            "unit_mixed": set(profile.get("unit_mixing", {})),
            "leak_candidates": leak_cols,
            "minority_share": (self.tinfo or {}).get("minority_share"),
            "prompted": self.prompted,
        }

    def record_events(self, events):
        with self.lock:
            for event in events:
                if not isinstance(event, dict) or "type" not in event:
                    continue
                self.log({"client": event})
                if event["type"] == "target_select" and self.df is not None:
                    self._set_target(event.get("column"))
                self.competence.observe(event, self._context())
            self.log({"type": "competence_snapshot", "heuristic": self.competence.estimates()})

    # ── Dataset ─────────────────────────────────────────────────────────────
    def set_dataset(self, df):
        with self.lock:
            self.df = df
            self.profile = guardrails.profile_dataset(df)
            self.semantic = guardrails.semantic_leak_candidates(
                self.dictionary, self.lang, self.decision_layer)
            self.tinfo = None
            self.log({"type": "dataset_profile", "profile": self.profile,
                      "semantic_leak_candidates": self.semantic})

    def _set_target(self, target):
        if self.df is None or not target or target not in self.df.columns:
            return
        if self.tinfo and self.tinfo.get("target") == target:
            return
        self.tinfo = guardrails.target_info(self.df, target)
        self.log({"type": "target_info", "target_info": self.tinfo})

    # ── Competence ──────────────────────────────────────────────────────────
    def current_estimates(self):
        heuristic = self.competence.estimates()
        decision = None
        if self.decision_layer is not None:
            decision = self.competence.decision_estimates(self.decision_layer)
        self.log({"type": "competence_estimate", "heuristic": heuristic, "decision_layer": decision})
        if self.competence_source == "decision_layer" and decision:
            return decision
        return heuristic

    def chat_instruction(self):
        if self.condition != "adaptive":
            return ""
        with self.lock:
            return profile_instruction(self.current_estimates())

    # ── Guidance ────────────────────────────────────────────────────────────
    def guidance(self, stage, state):
        with self.lock:
            if self.profile is None:
                return []
            self._set_target(state.get("target"))
            issues = guardrails.evaluate(stage, state, self.profile, self.tinfo, self.semantic)
            estimates = self.current_estimates()
            out = []
            for issue in issues:
                intervention = self._decide(issue, estimates, stage)
                if intervention is None:
                    continue
                marker = (issue["key"], intervention["mode"], stage)
                intervention["new"] = marker not in self.shown
                self.shown.add(marker)
                out.append(intervention)
            self.log({"type": "guidance", "stage": stage, "state": state,
                      "issues": issues, "estimates": estimates, "interventions": out})
            return out

    def _decide(self, issue, estimates, stage):
        if self.condition == "baseline":
            return None
        kind = issue["type"]
        if self.condition == "generic":
            mode, level = "inform", "intermediate"
            blocking = stage == "pre_train" and issue["severity"] == "high"
        else:
            est = estimates[issue["dimension"]]
            if est["confidence"] < ASK_CONFIDENCE and issue["key"] not in self.probed:
                mode, level, blocking = "ask", None, stage == "pre_train"
            elif est["score"] < NOVICE_BELOW:
                mode, level, blocking = "inform", "novice", stage == "pre_train"
            elif est["score"] <= EXPERT_ABOVE:
                mode, level, blocking = "inform", "intermediate", False
            else:
                mode, level, blocking = "subtle", "expert", False

        column = issue.get("column")
        values = format_facts(issue["facts"], column=column,
                              description=self._describe(column), moment=self._moment(),
                              lang=self.lang)
        intervention = {
            "key": issue["key"], "type": kind, "dimension": issue["dimension"],
            "column": column, "stage": stage, "mode": mode, "level": level,
            "blocking": blocking, "title": render_title(kind, self.lang),
        }
        if mode == "ask":
            intervention["probe"] = render_probe(kind, self.lang, values)
        else:
            intervention["text"] = render_text(kind, level, self.lang, values, stage)
            self.prompted.add(issue["key"])
            self.prompted.add(kind)
            if column:
                self.prompted.add(column)
        return intervention

    def answer_probe(self, key, answer):
        """Record the answer to a probe; returns whether it was correct."""
        with self.lock:
            kind = key.split(":", 1)[0]
            if kind not in guardrails.ISSUE_DIMENSION:
                raise ValueError("unknown issue")
            self.probed.add(key)
            correct = answer == "correct"
            event = {"type": "guidance_answer", "key": key, "answer": answer,
                     "correct": correct, "dimension": guardrails.ISSUE_DIMENSION[kind],
                     "ts": round(time.time() * 1000)}
            self.log({"client": event})
            self.competence.observe(event, self._context())
            return correct

    def _describe(self, column):
        if not column or not self.dictionary:
            return ""
        desc = self.dictionary.get("columns", {}).get(column, {})
        return desc.get(self.lang) or desc.get("en") or ""

    def _moment(self):
        if not self.dictionary:
            return {"es": "en el momento de la predicción",
                    "en": "at prediction time"}.get(self.lang, "at prediction time")
        moment = self.dictionary.get("prediction_moment", {})
        return moment.get(self.lang) or moment.get("en") or ""


class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()
        self.decision_layer = TypedDecisionLayer.from_env()
        self.competence_source = os.getenv("COMPETENCE_SOURCE", "heuristic")

    def create(self, participant_id, condition, dataset_key=None, lang="es"):
        session = StudySession(participant_id, condition, dataset_key, lang,
                               self.decision_layer, self.competence_source)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)


store = SessionStore()
