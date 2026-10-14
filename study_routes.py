"""Flask routes for the user study (telemetry, guidance, task submission).

All routes are inert unless the frontend was opened in study mode, e.g.
    /?study=1&pid=P07&cond=adaptive&ds=A&lang=es
"""
import logging
import traceback

from flask import Blueprint, jsonify, request

from utils.study import CONDITIONS, store

logger = logging.getLogger(__name__)
study_bp = Blueprint("study", __name__, url_prefix="/study")

GUIDANCE_STAGES = ("data", "pre_train", "post_train")
PROBE_ANSWERS = ("correct", "wrong", "dont_know")


def _read_table(file):
    """Load an uploaded table; CSV is handled here to keep the study light."""
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(file)
    from utils.model_utils import load_dataframe
    return load_dataframe(file)


def _session_or_404(session_id):
    session = store.get(session_id or "")
    if session is None:
        return None, (jsonify({"error": "unknown_session"}), 404)
    return session, None


@study_bp.route("/session", methods=["POST"])
def create_session():
    data = request.get_json(silent=True) or {}
    condition = data.get("condition", "baseline")
    if condition not in CONDITIONS:
        return jsonify({"error": "unknown_condition"}), 400
    session = store.create(data.get("participant_id"), condition,
                           data.get("dataset_key"), data.get("lang", "es"))
    return jsonify({"session_id": session.id, "condition": session.condition,
                    "dictionary": session.dictionary})


@study_bp.route("/telemetry", methods=["POST"])
def telemetry():
    data = request.get_json(silent=True) or {}
    session, error = _session_or_404(data.get("session_id"))
    if error:
        return error
    events = data.get("events") or []
    if not isinstance(events, list):
        return jsonify({"error": "invalid_events"}), 400
    session.record_events(events[:500])
    return jsonify({"ok": True})


@study_bp.route("/dataset", methods=["POST"])
def dataset():
    session, error = _session_or_404(request.form.get("session_id"))
    if error:
        return error
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "error_no_file"}), 400
    try:
        session.set_dataset(_read_table(file))
    except Exception:
        logger.error(f"Study dataset profiling failed: {traceback.format_exc()}")
        return jsonify({"error": "profile_failed"}), 500
    return jsonify({"ok": True})


@study_bp.route("/guidance", methods=["POST"])
def guidance():
    data = request.get_json(silent=True) or {}
    session, error = _session_or_404(data.get("session_id"))
    if error:
        return error
    stage = data.get("stage")
    if stage not in GUIDANCE_STAGES:
        return jsonify({"error": "invalid_stage"}), 400
    try:
        interventions = session.guidance(stage, data.get("state") or {})
    except Exception:
        logger.error(f"Study guidance failed: {traceback.format_exc()}")
        return jsonify({"interventions": []})
    return jsonify({"interventions": interventions})


@study_bp.route("/probe", methods=["POST"])
def probe():
    data = request.get_json(silent=True) or {}
    session, error = _session_or_404(data.get("session_id"))
    if error:
        return error
    answer = data.get("answer")
    if answer not in PROBE_ANSWERS:
        return jsonify({"error": "invalid_answer"}), 400
    try:
        correct = session.answer_probe(str(data.get("key", "")), answer)
    except ValueError:
        return jsonify({"error": "unknown_issue"}), 400
    return jsonify({"correct": correct})
