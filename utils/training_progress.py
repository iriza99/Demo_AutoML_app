"""Real training progress, reported by AutoGluon itself.

The frontend used to animate a fake model queue on a timer. Instead, a
callback receives AutoGluon's own before/after-model-fit events, so the
progress panel only shows what has actually happened: which model is being
fitted, which ones finished (with their validation score) and which failed.
The same callback also makes "stop" effective between models.
"""
import os
import threading
import time

try:  # AutoGluon is optional for the unit tests
    from autogluon.core.callbacks import AbstractCallback
except ImportError:  # pragma: no cover
    AbstractCallback = object


class TrainingProgress:
    """Thread-safe progress record for one training run."""

    def __init__(self, time_limit=None):
        self._lock = threading.Lock()
        self.started = time.time()
        self.time_limit = time_limit
        self.phase = "preparing"      # preparing → fitting → evaluating → done
        self.models = []              # dicts: name, status, score, fit_time
        self.metric = None
        self.gpu = False
        self.stopping = False

    def set_phase(self, phase):
        with self._lock:
            self.phase = phase

    def model_started(self, name, ensemble=False):
        with self._lock:
            self.phase = "fitting"
            self.models.append({"name": name, "status": "running", "score": None,
                                "fit_time": None, "ensemble": ensemble, "started": time.time()})

    def model_finished(self, name, ok, score=None, fit_time=None, status=None):
        with self._lock:
            for m in reversed(self.models):
                if m["name"] == name and m["status"] == "running":
                    m["status"] = status or ("done" if ok else "failed")
                    m["score"] = score
                    m["fit_time"] = fit_time if fit_time is not None else round(time.time() - m["started"], 1)
                    break

    def to_dict(self):
        with self._lock:
            return {
                "phase": self.phase,
                "elapsed": round(time.time() - self.started, 1),
                "time_limit": self.time_limit,
                "metric": self.metric,
                "gpu": self.gpu,
                "stopping": self.stopping,
                "models": [{k: v for k, v in m.items() if k != "started"} for m in self.models],
            }


# AutoGluon pickles the trainer (callbacks included) during fit, and locks
# cannot be pickled, so callbacks only keep a key into this registry.
_RUNS = {}
_RUNS_LOCK = threading.Lock()


class ProgressCallback(AbstractCallback):
    """AutoGluon callback that feeds a TrainingProgress and honours a stop event."""

    def __init__(self, progress, stop_event=None):
        super().__init__()
        self.run_key = f"{id(progress)}-{time.time()}"
        with _RUNS_LOCK:
            _RUNS[self.run_key] = (progress, stop_event)
        self._current = None

    @property
    def progress(self):
        return _RUNS.get(self.run_key, (_NULL_PROGRESS, None))[0]

    @property
    def stop_event(self):
        return _RUNS.get(self.run_key, (None, None))[1]

    def release(self):
        """Forget the live objects once training is over."""
        with _RUNS_LOCK:
            _RUNS.pop(self.run_key, None)

    def _stopped(self):
        return bool(self.stop_event and self.stop_event.is_set())

    def _before_model_fit(self, trainer, model, time_limit=None, stack_name="core", level=1, **kwargs):
        if self._stopped():
            self.progress.stopping = True
            return True, False  # end the trainer fit now
        if self.progress.metric is None:
            metric = getattr(trainer, "eval_metric", None)
            self.progress.metric = getattr(metric, "name", None)
        self._current = model.name
        self.progress.model_started(model.name, ensemble=stack_name != "core")
        return False, False

    def _after_model_fit(self, trainer, model_names, stack_name="core", level=1, **kwargs):
        name = self._current
        if model_names:
            fitted = model_names[0]
            score = fit_time = None
            try:
                score = trainer.get_model_attribute(fitted, "val_score")
                fit_time = trainer.get_model_attribute(fitted, "fit_time")
                metric = getattr(trainer, "eval_metric", None)
                # AutoGluon stores errors negated (higher is better); show them as errors
                if score is not None and metric is not None and not getattr(metric, "greater_is_better", True):
                    score = -score
            except Exception:
                pass
            self.progress.model_finished(name, True, score=_round(score), fit_time=_round(fit_time, 1))
        else:
            self.progress.model_finished(name, False, status=_failure_status(trainer, name))
        if self._stopped():
            self.progress.stopping = True
            return True
        return False


class _NullProgress:
    """Stand-in once a run is released (e.g. a reloaded predictor refits)."""
    stopping = False
    metric = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


_NULL_PROGRESS = _NullProgress()


def _failure_status(trainer, name):
    """Why a model produced nothing: missing library, no time left, or a crash."""
    errors = getattr(trainer, "_models_failed_to_train_errors", {}) or {}
    info = errors.get(name)
    if info is None:
        return "skipped"  # AutoGluon skipped it, e.g. not enough time remaining
    if info.get("exc_type") in ("ImportError", "ModuleNotFoundError"):
        return "unavailable"
    if info.get("exc_type") == "TimeLimitExceeded":
        return "skipped"
    return "failed"


def _round(value, digits=4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def gpu_available():
    """Whether to train on GPU: AUTOML_GPU=auto (default) | 1 | 0."""
    if os.getenv("AUTOML_GPU", "auto").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_hyperparameters():
    """AutoGluon's default model set, with GPU enabled where it is reliable.

    XGBoost and CatBoost ship GPU support in their standard wheels; the
    PyTorch/FastAI networks use the GPU automatically once fit() is given
    one. LightGBM stays on CPU because its pip wheel has no CUDA build.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    hp = get_hyperparameter_config("default")
    for key in ("XGB", "CAT"):
        configs = hp.get(key)
        if configs is None:
            continue
        for cfg in configs if isinstance(configs, list) else [configs]:
            cfg.setdefault("ag_args_fit", {})["num_gpus"] = 1
    return hp

