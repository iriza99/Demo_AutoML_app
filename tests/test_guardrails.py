import json
import os

import numpy as np
import pandas as pd
import pytest

from utils import guardrails

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("hf_readmission", "readmission_30d"), ("ed_sepsis", "sepsis_48h")]


def _load(name):
    df = pd.read_csv(os.path.join(ROOT, "study", "datasets", f"{name}.csv"))
    with open(os.path.join(ROOT, "study", "datasets", f"{name}_dictionary.json"), encoding="utf-8") as fh:
        dictionary = json.load(fh)
    with open(os.path.join(ROOT, "study", "answer_key", f"{name}_traps.json")) as fh:
        traps = {t["id"]: t for t in json.load(fh)["traps"]}
    return df, dictionary, traps


@pytest.mark.parametrize("name,target", DATASETS)
def test_detectors_find_every_planted_trap_and_nothing_else(name, target):
    df, dictionary, traps = _load(name)
    profile = guardrails.profile_dataset(df)
    tinfo = guardrails.target_info(df, target)
    semantic = guardrails.semantic_leak_candidates(dictionary, "es")

    assert profile["identifiers"] == [traps["M2"]["column"]]
    assert list(profile["unit_mixing"]) == [traps["C2"]["column"]]
    assert abs(profile["unit_mixing"][traps["C2"]["column"]]["factor"] - traps["C2"]["factor"]) < 1e-6
    assert list(semantic) == [traps["C1"]["column"]]

    issues = guardrails.evaluate("pre_train", {"target": target, "metric": "accuracy"},
                                 profile, tinfo, semantic)
    keys = {i["key"] for i in issues}
    assert keys == {
        "imbalance_metric",
        f"identifier:{traps['M2']['column']}",
        f"temporal_leak:{traps['C1']['column']}",
        f"unit_mixing:{traps['C2']['column']}",
    }


@pytest.mark.parametrize("name,target", DATASETS)
def test_fixing_the_configuration_closes_the_issues(name, target):
    df, dictionary, traps = _load(name)
    profile = guardrails.profile_dataset(df)
    tinfo = guardrails.target_info(df, target)
    semantic = guardrails.semantic_leak_candidates(dictionary, "es")
    state = {"target": target, "metric": "roc_auc",
             "dropped": [traps[k]["column"] for k in ("M2", "C1", "C2")]}
    assert guardrails.evaluate("pre_train", state, profile, tinfo, semantic) == []


def test_data_stage_only_reports_target_independent_issues():
    df, dictionary, traps = _load("hf_readmission")
    profile = guardrails.profile_dataset(df)
    issues = guardrails.evaluate("data", {}, profile, None, {})
    assert [i["type"] for i in issues] == ["unit_mixing"]


def test_post_train_flags_dominant_identifier():
    df, dictionary, traps = _load("ed_sepsis")
    profile = guardrails.profile_dataset(df)
    tinfo = guardrails.target_info(df, "sepsis_48h")
    importance = {traps["M2"]["column"]: 0.2, "lactate": 0.05, "age": 0.03}
    issues = guardrails.evaluate("post_train", {"target": "sepsis_48h", "metric": "roc_auc",
                                                "dropped": [traps["C1"]["column"]],
                                                "feature_importance": importance},
                                 profile, tinfo, {})
    assert [i["key"] for i in issues] == ["unit_mixing:lactate", f"identifier:{traps['M2']['column']}"]
    assert issues[1]["facts"]["importance_rank"] == 1


def test_natural_skewed_and_normal_variables_are_not_unit_mixed():
    rng = np.random.default_rng(0)
    assert guardrails.detect_unit_mixing(pd.Series(np.exp(rng.normal(8, 0.9, 2000)))) is None
    assert guardrails.detect_unit_mixing(pd.Series(rng.normal(120, 20, 2000))) is None
    assert guardrails.detect_unit_mixing(pd.Series(rng.poisson(3, 2000))) is None


def test_glucose_unit_mixing_is_detected():
    rng = np.random.default_rng(1)
    mmol = np.exp(rng.normal(np.log(6.5), 0.25, 1500))
    mixed = np.where(rng.random(1500) < 0.3, mmol * 18.0, mmol)
    found = guardrails.detect_unit_mixing(pd.Series(mixed))
    assert found and found["factor"] == 18.0


def test_identifier_detection_needs_unique_consecutive_values():
    assert guardrails.is_identifier(pd.Series(range(1000, 1500)), "row")
    assert guardrails.is_identifier(pd.Series(np.random.default_rng(2).permutation(800) + 1), "anything")
    assert not guardrails.is_identifier(pd.Series(np.random.default_rng(3).integers(40, 90, 500)), "age")
