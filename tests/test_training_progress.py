import pytest

pytest.importorskip("autogluon.core")

import threading

from utils.training_progress import ProgressCallback, TrainingProgress


class FakeMetric:
    name = "roc_auc"
    greater_is_better = True


class FakeTrainer:
    eval_metric = FakeMetric()
    attrs = {"LightGBM_BAG_L1": {"val_score": 0.91234, "fit_time": 2.345}}

    def get_model_attribute(self, name, attr):
        return self.attrs[name][attr]


class FakeModel:
    def __init__(self, name):
        self.name = name


def test_progress_reflects_real_model_events():
    progress = TrainingProgress(time_limit=60)
    cb = ProgressCallback(progress)
    trainer = FakeTrainer()
    assert cb.before_model_fit(trainer, FakeModel("LightGBM_BAG_L1")) == (False, False)
    assert progress.to_dict()["models"][0]["status"] == "running"
    assert cb.after_model_fit(trainer, ["LightGBM_BAG_L1"]) is False
    cb.before_model_fit(trainer, FakeModel("CatBoost_BAG_L1"))
    trainer._models_failed_to_train_errors = {"CatBoost_BAG_L1": {"exc_type": "ImportError"}}
    cb.after_model_fit(trainer, [])  # library missing
    state = progress.to_dict()
    assert state["phase"] == "fitting" and state["metric"] == "roc_auc"
    assert state["models"][0] == {"name": "LightGBM_BAG_L1", "status": "done", "score": 0.9123,
                                  "fit_time": 2.3, "ensemble": False}
    assert state["models"][1]["status"] == "unavailable"
    cb.before_model_fit(trainer, FakeModel("XGBoost_BAG_L1"))
    cb.after_model_fit(trainer, [])  # skipped: not in the failure record
    assert progress.to_dict()["models"][2]["status"] == "skipped"


def test_error_metrics_are_shown_as_errors():
    class RMSE:
        name = "root_mean_squared_error"
        greater_is_better = False
    trainer = FakeTrainer()
    trainer.eval_metric = RMSE()
    trainer.attrs = {"XGBoost_BAG_L1": {"val_score": -3.5, "fit_time": 1.0}}
    progress = TrainingProgress()
    cb = ProgressCallback(progress)
    cb.before_model_fit(trainer, FakeModel("XGBoost_BAG_L1"))
    cb.after_model_fit(trainer, ["XGBoost_BAG_L1"])
    assert progress.to_dict()["models"][0]["score"] == 3.5


def test_stop_event_ends_training_between_models():
    stop = threading.Event()
    progress = TrainingProgress()
    cb = ProgressCallback(progress, stop)
    trainer = FakeTrainer()
    cb.before_model_fit(trainer, FakeModel("LightGBM_BAG_L1"))
    stop.set()
    assert cb.after_model_fit(trainer, ["LightGBM_BAG_L1"]) is True
    assert cb.before_model_fit(trainer, FakeModel("CatBoost_BAG_L1")) == (True, False)
    assert progress.to_dict()["stopping"] is True
