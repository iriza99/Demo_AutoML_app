import io
import json
import os

import pandas as pd
import pytest
from flask import Flask

import utils.study as study_module
from study_routes import study_bp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_A = os.path.join(ROOT, "study", "datasets", "hf_readmission.csv")


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(study_module, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(study_module.store, "decision_layer", None)
    return tmp_path


def _session(condition):
    session = study_module.store.create("P01", condition, "A", "es")
    session.set_dataset(pd.read_csv(CSV_A))
    return session


PRE_TRAIN = {"target": "readmission_30d", "metric": "accuracy", "dropped": []}


def test_baseline_shows_nothing_but_logs(tmp_logs):
    session = _session("baseline")
    assert session.guidance("pre_train", PRE_TRAIN) == []
    lines = [json.loads(l) for l in open(session.log_path)]
    assert {"session_created", "dataset_profile", "guidance"} <= {l.get("type") for l in lines}


def test_generic_shows_every_issue_identically():
    interventions = _session("generic").guidance("pre_train", PRE_TRAIN)
    assert len(interventions) == 4
    assert {(i["mode"], i["level"], i["blocking"]) for i in interventions} == {("inform", "intermediate", True)}


def test_adaptive_probes_when_unsure_then_tailors():
    session = _session("adaptive")
    first = session.guidance("pre_train", PRE_TRAIN)
    assert {i["mode"] for i in first} == {"ask"}
    for iv in first:
        assert iv["probe"]["options"]
    # An ML expert who misses the clinical traps
    session.answer_probe("imbalance_metric", "correct")
    session.answer_probe("identifier:record_id", "correct")
    session.answer_probe("temporal_leak:nt_probnp_followup", "wrong")
    session.answer_probe("unit_mixing:creatinine", "dont_know")
    second = {i["type"]: i for i in session.guidance("pre_train", PRE_TRAIN)}
    assert second["imbalance_metric"]["mode"] == "subtle"
    assert second["identifier"]["mode"] == "subtle"
    assert second["temporal_leak"]["level"] == "novice" and second["temporal_leak"]["blocking"]
    assert second["unit_mixing"]["level"] == "novice"
    assert "alta hospitalaria" in second["temporal_leak"]["text"]


def test_chat_instruction_only_in_adaptive():
    session = _session("adaptive")
    session.answer_probe("identifier:record_id", "wrong")
    assert "little machine-learning knowledge" in session.chat_instruction()
    generic = _session("generic")
    generic.answer_probe("identifier:record_id", "wrong")
    assert generic.chat_instruction() == ""


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(study_bp)
    return app.test_client()


def test_routes_end_to_end(client):
    res = client.post("/study/session", json={"participant_id": "P/../02", "condition": "adaptive",
                                              "dataset_key": "A", "lang": "en"})
    data = res.get_json()
    sid = data["session_id"]
    assert data["dictionary"]["target"] == "readmission_30d"
    assert study_module.store.get(sid).participant_id == "P02"

    with open(CSV_A, "rb") as fh:
        res = client.post("/study/dataset", data={"session_id": sid,
                                                  "file": (io.BytesIO(fh.read()), "hf.csv")})
    assert res.status_code == 200

    res = client.post("/study/telemetry", json={"session_id": sid, "events": [
        {"type": "target_select", "column": "readmission_30d", "ts": 1},
        {"type": "metric_change", "metric": "roc_auc", "ts": 2},
    ]})
    assert res.get_json() == {"ok": True}

    res = client.post("/study/guidance", json={"session_id": sid, "stage": "data", "state": {}})
    interventions = res.get_json()["interventions"]
    assert [i["type"] for i in interventions] == ["unit_mixing"]

    res = client.post("/study/probe", json={"session_id": sid, "key": "unit_mixing:creatinine",
                                            "answer": "correct"})
    assert res.get_json() == {"correct": True}


def test_routes_validate_input(client):
    assert client.post("/study/session", json={"condition": "magic"}).status_code == 400
    assert client.post("/study/telemetry", json={"session_id": "nope"}).status_code == 404
    sid = client.post("/study/session", json={"condition": "generic"}).get_json()["session_id"]
    assert client.post("/study/guidance", json={"session_id": sid, "stage": "x"}).status_code == 400
    assert client.post("/study/probe", json={"session_id": sid, "key": "x", "answer": "maybe"}).status_code == 400
    assert client.post("/study/probe", json={"session_id": sid, "key": "bogus:x", "answer": "correct"}).status_code == 400
    # No dataset uploaded yet: guidance is empty rather than an error
    assert client.post("/study/guidance", json={"session_id": sid, "stage": "pre_train"}).get_json() == {"interventions": []}
