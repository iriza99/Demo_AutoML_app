import pytest

pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")

from utils.report import build_report  # noqa: E402

STATS = [
    {"name": "age", "type": "Numeric", "missing": 0, "unique": 40, "min": 20, "max": 90, "mean": 55.2, "std": 12.1},
    {"name": "sex", "type": "Categorical", "missing": 2, "unique": 2, "top": [["F", 50], ["M", 48]]},
    {"name": "admitted", "type": "Date", "missing": 0, "unique": 90},
    {"name": "outcome", "type": "Categorical", "missing": 0, "unique": 2},
]


def _pdf(data, lang):
    out = build_report(data, lang=lang).getvalue()
    assert out.startswith(b"%PDF") and len(out) > 5000
    return out


@pytest.mark.parametrize("lang", ["es", "en", "fr", "xx"])
def test_classification_report(lang):
    _pdf({
        "dataset_name": "demo.csv", "target_column": "outcome", "n_rows": 100, "n_columns": 4,
        "excluded": ["record_id"], "imputation": {"sex": {"method": "constant", "constant": "U"}},
        "task_type": "binary", "best_model": "LightGBM", "train_time": 12.3, "used_gpu": False,
        "metrics": {"roc_auc": 0.81, "accuracy": 0.77},
        "plot_data": {"labels": ["no", "yes"], "confusion": [[40, 10], [13, 37]],
                      "roc": {"fpr": [0, 0.2, 1], "tpr": [0, 0.7, 1], "threshold": [None, 0.5, 0], "auc": 0.81,
                              "positive_label": "yes"}},
        "feature_importance": {"age": 0.1, "sex": -0.01},
        "leaderboard": [{"model": "LightGBM", "score_val": 0.81, "fit_time": 3.2}, {"model": "KNN", "score_val": None}],
        "stats": STATS,
    }, lang)


def test_regression_and_minimal_reports():
    _pdf({"task_type": "regression", "metrics": {"r2": 0.6, "rmse": 3.1},
          "plot_data": {"y_true": [1, 2, 3, 4], "y_pred": [1.2, 1.9, 3.3, 3.8],
                        "error_hist": {"counts": [1, 2, 1], "edges": [-0.3, -0.1, 0.1, 0.3]}, "n": 4},
          "stats": STATS}, "es")
    # Whatever is missing, a report is still produced
    build_report({}, lang="en")
    build_report({"metrics": {"rmse": float("nan")}, "stats": [{"name": None}], "leaderboard": "x"}, lang="en")
