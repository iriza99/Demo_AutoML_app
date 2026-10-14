from utils.competence import CompetenceModel

CONTEXT = {"identifiers": {"record_id"}, "leak_candidates": {"nt_probnp_followup"},
           "unit_mixed": {"creatinine"}, "minority_share": 0.12, "prompted": set()}


def test_starts_uncertain():
    est = CompetenceModel().estimate("ml")
    assert est["score"] == 2.0 and est["confidence"] == 0.0


def test_self_initiated_ml_actions_raise_ml_not_domain():
    model = CompetenceModel()
    model.observe({"type": "metric_change", "metric": "roc_auc"}, CONTEXT)
    model.observe({"type": "column_drop_toggle", "column": "record_id", "dropped": True}, CONTEXT)
    assert model.estimate("ml")["score"] > 2.5
    assert model.estimate("domain")["score"] == 2.0


def test_actions_after_a_prompt_are_not_evidence():
    model = CompetenceModel()
    ctx = dict(CONTEXT, prompted={"nt_probnp_followup", "imbalance_metric"})
    model.observe({"type": "column_drop_toggle", "column": "nt_probnp_followup", "dropped": True}, ctx)
    model.observe({"type": "metric_change", "metric": "roc_auc"}, ctx)
    assert model.estimate("domain")["score"] == 2.0
    assert model.estimate("ml")["score"] == 2.0


def test_dropping_leak_is_domain_evidence():
    model = CompetenceModel()
    model.observe({"type": "column_drop_toggle", "column": "nt_probnp_followup", "dropped": True}, CONTEXT)
    assert model.estimate("domain")["score"] > 2.4


def test_chat_vocabulary_and_basic_questions():
    expert = CompetenceModel()
    expert.observe({"type": "chat_send", "text": "Is the AUC estimated out-of-fold? I worry about overfitting"}, CONTEXT)
    novice = CompetenceModel()
    novice.observe({"type": "chat_send", "text": "¿Qué es el AUC?"}, CONTEXT)
    clinician = CompetenceModel()
    clinician.observe({"type": "chat_send", "text": "La creatinina de algunos pacientes está en µmol/L"}, CONTEXT)
    assert expert.estimate("ml")["score"] > 2.0 > novice.estimate("ml")["score"]
    assert clinician.estimate("domain")["score"] > 2.0


def test_probe_answers():
    model = CompetenceModel()
    model.observe({"type": "guidance_answer", "dimension": "domain", "correct": True, "key": "x"}, CONTEXT)
    model.observe({"type": "guidance_answer", "dimension": "ml", "correct": False, "answer": "wrong", "key": "y"}, CONTEXT)
    assert model.estimate("domain")["score"] > 2.4
    assert model.estimate("ml")["score"] < 1.7


def test_training_with_pitfalls_lowers_estimates():
    model = CompetenceModel()
    model.observe({"type": "train_start", "config": {"dropped": [], "metric": "accuracy"}}, CONTEXT)
    assert model.estimate("ml")["score"] < 2.0
    assert model.estimate("domain")["score"] < 2.0


def test_summary_is_serialisable():
    import json
    model = CompetenceModel()
    model.observe({"type": "step_view", "step": 1, "ts": 0}, CONTEXT)
    model.observe({"type": "step_view", "step": 2, "ts": 30000}, CONTEXT)
    model.observe({"type": "chat_send", "text": "hola"}, CONTEXT)
    summary = model.summary()
    assert summary["seconds_per_wizard_step"] == {"1": 30}
    json.dumps(summary)
