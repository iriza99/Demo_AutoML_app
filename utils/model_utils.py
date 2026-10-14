from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
import io
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import logging
import traceback

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np
import shap
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# Load a dataset from a file path or file-like object into a DataFrame.
# Accepts .csv, .xls, .xlsx, .xlsm, and .arff files.
def load_dataframe(source):
    if hasattr(source, 'filename'):
        # Flask FileStorage: read bytes, detect extension from filename
        filename = source.filename
        ext = os.path.splitext(filename)[1].lower()
        raw = source.read()
        if ext == '.csv':
            return pd.read_csv(io.BytesIO(raw))
        elif ext in ('.xls', '.xlsx', '.xlsm'):
            return pd.read_excel(io.BytesIO(raw))
        elif ext == '.arff':
            from scipy.io import arff as scipy_arff
            data, _ = scipy_arff.loadarff(io.StringIO(raw.decode('utf-8')))
            df = pd.DataFrame(data)
        else:
            raise UserError("error_unsupported_file_format")
    else:
        # File path string
        ext = os.path.splitext(source)[1].lower()
        if ext == '.csv':
            return pd.read_csv(source)
        elif ext in ('.xls', '.xlsx', '.xlsm'):
            return pd.read_excel(source)
        elif ext == '.arff':
            from scipy.io import arff as scipy_arff
            data, _ = scipy_arff.loadarff(source)
            df = pd.DataFrame(data)
        else:
            raise UserError("error_unsupported_file_format")

    # Decode byte strings produced by scipy arff loader
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df


# Cache for SHAP explainers keyed by model_path to avoid rebuilding on every request
_shap_explainer_cache: dict = {}

# Custom exception for user errors
class UserError(Exception):
    def __init__(self, message_key, status_code=400):
        super().__init__(message_key)
        self.message_key = message_key
        self.status_code = status_code

# Save a matplotlib plot to disk
def save_plot_to_disk(plt_obj, save_path, filename):
    full_path = os.path.join(save_path, filename)
    plt_obj.savefig(full_path, format='png', bbox_inches='tight')
    return full_path

# Convert a SHAP plot to a base64-encoded PNG image
def shap_plot_to_base64(plot_func, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs, show=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

# Generate a histogram of prediction errors (RMSE) and return as base64 image
def generate_rmse_error_histogram_base64(y_true, y_pred, save_dir=None):
    errors = y_pred - y_true  
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='orange', stat='density')
    plt.axvline(0, color='blue', linestyle='--', label='Zero error')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    if save_dir:
        save_plot_to_disk(plt, save_dir, "rmse_error_distribution.png")

    # Encode as base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

# Generate a feature importance bar plot and return as base64 image
def generate_feature_importance_plot(importances, save_dir=None):
    # Keep only features with non-trivial importance; fall back to top-20 if all near zero
    significant = importances[importances['importance'].abs() > 1e-6]
    importances = (significant if not significant.empty else importances).head(20)
    n = len(importances)
    fig_height = max(4, n * 0.35)
    plt.figure(figsize=(10, fig_height))

    all_near_zero = importances['importance'].abs().max() < 1e-4
    if all_near_zero:
        plt.text(0.5, 0.5, 'Feature importance not available\n(all values ≈ 0)',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=13, color='gray')
        plt.axis('off')
    else:
        sns.barplot(x=importances['importance'], y=importances.index, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.tight_layout()

    if save_dir:
        save_plot_to_disk(plt, save_dir, "feature_importance.png")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

# Generate metric-specific plots (confusion matrix, ROC, F1, RMSE) and return as base64 image
def generate_metric_plot(metric_name, y_true=None, y_pred=None, y_proba=None, class_labels=None, save_dir=None):
    fig, ax = plt.subplots()
    metric_name = metric_name.lower()
    filename = f"{metric_name}.png"

    if metric_name == 'accuracy':
        # Display confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title("Confusion Matrix")

    elif metric_name == 'roc_auc' and y_proba is not None:
        pos_label = class_labels[1] if class_labels else 1
        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
        ax.plot(fpr, tpr, label='ROC Curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

    elif metric_name == 'f1_macro':
        # F1 score bar plot per class
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=class_labels)
        ax.bar(class_labels, f1_per_class, color='lightgreen')
        ax.set_title("F1 Score per Class")
        ax.set_ylabel("F1 Score")

    elif metric_name == 'rmse':
        # Scatter plot for regression predictions vs actual values
        ax.scatter(y_true, y_pred, alpha=0.5, color='coral', label='Predictions')
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        rmse_value = mean_squared_error(y_true, y_pred) ** 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predictions vs Actual (RMSE = {rmse_value:.4f})")
        ax.legend()

    else:
        ax.text(0.5, 0.5, f"No plot available for {metric_name}", ha='center', va='center')
        ax.set_axis_off()

    plt.tight_layout()
    if save_dir:
        save_plot_to_disk(plt, save_dir, filename)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

# Log exactly which Python and AutoGluon are being used by the Flask server
import sys
logging.info(f"Python executable: {sys.executable}")
try:
    import autogluon.tabular as _ag_tab
    logging.info(f"AutoGluon tabular version: {_ag_tab.__version__}, location: {_ag_tab.__file__}")
    logging.info(f"TabularPredictor has get_oof_pred: {hasattr(_ag_tab.TabularPredictor, 'get_oof_pred')}")
except Exception as _e:
    logging.warning(f"AutoGluon version check failed: {_e}")

# Train an AutoGluon model and return training results and plots
def train_model(df, target_column, time_limit, save_path, stop_event=None, time_column=None,
                prediction_length_percent=None, eval_metric=None, excluded_model_types=None,
                num_bag_folds=None, val_size=None):
    try:
        save_path = os.path.abspath(save_path)

        plot_dir = os.path.join(save_path, "plot_train_results")
        os.makedirs(plot_dir, exist_ok=True)

        if df[target_column].isnull().any():
            raise UserError("error_missing_target_values")

        # Time series branch
        if time_column:
            return train_model_timeseries(
                df, target_column, time_column, time_limit, save_path, stop_event, plot_dir,
                prediction_length_percent, eval_metric=eval_metric, excluded_model_types=excluded_model_types
            )

        # When bagging is disabled, split off a held-out validation set.
        # AutoGluon trains on df_train; metrics and plots come from df_val (unseen).
        df_val = None
        df_train = df  # keep reference to training data for feature importance
        if not num_bag_folds and val_size:
            from sklearn.model_selection import train_test_split
            df_train, df_val = train_test_split(
                df, test_size=val_size, random_state=42,
                stratify=df[target_column] if df[target_column].nunique() < 50 else None
            )
            logging.info(f"Validation split: {len(df_train)} train / {len(df_val)} val ({val_size*100:.0f}%)")
            df = df_train  # train AutoGluon on the smaller set

        predictor = TabularPredictor(label=target_column, path=save_path, eval_metric=eval_metric)

        if stop_event and stop_event.is_set():
            raise UserError("training_interrupted")

        fit_kwargs = {'time_limit': time_limit}
        if excluded_model_types:
            fit_kwargs['excluded_model_types'] = excluded_model_types
        if num_bag_folds is not None:
            fit_kwargs['num_bag_folds'] = num_bag_folds
        if num_bag_folds:
            # All rows participate in OOF evaluation — no holdout withheld.
            fit_kwargs['use_bag_holdout'] = False
        if df_val is not None:
            # Tell AutoGluon to use the same val split for its own model selection,
            # so leaderboard score_val and our metrics are on identical data.
            fit_kwargs['tuning_data'] = df_val
        try:
            predictor.fit(df, **fit_kwargs)
        except ValueError as ve:
            if eval_metric and 'not a valid metric' in str(ve):
                logging.warning(f"eval_metric '{eval_metric}' incompatible with detected task type, retrying without it.")
                predictor = TabularPredictor(label=target_column, path=save_path)
                fit_kwargs_retry = {k: v for k, v in fit_kwargs.items()}
                predictor.fit(df, **fit_kwargs_retry)
            else:
                raise

        # ── Capture OOF predictions right after fit() ────────────────────────────
        # Used ONLY for plots (confusion matrix, ROC, F1 bar, RMSE scatter).
        # Scalar metric values come from leaderboard(extra_metrics=...) instead.
        _oof_pred = None        # predicted class labels (or regression values)
        _oof_pred_proba = None  # probability DataFrame (classification only)

        pt = predictor.problem_type

        # predict_proba_oof → probability DataFrame → derive labels via argmax
        _proba_method = getattr(predictor, 'predict_proba_oof', None)
        if _proba_method is not None:
            for _mkw in [{}, {'silent': True}]:
                try:
                    _raw = _proba_method(**_mkw)
                    if pt == 'regression':
                        _oof_pred = _raw
                    elif isinstance(_raw, pd.DataFrame):
                        _oof_pred_proba = _raw
                        _oof_pred = _raw.idxmax(axis=1)
                    else:  # binary Series of positive-class probabilities
                        _oof_pred_proba = _raw
                        pos = predictor.class_labels[1]
                        neg = predictor.class_labels[0]
                        _oof_pred = _raw.map(lambda p: pos if p >= 0.5 else neg)
                    logging.info(f"OOF OK via predict_proba_oof({list(_mkw)}): {len(_oof_pred)} rows")
                    break
                except Exception as _e:
                    logging.warning(f"predict_proba_oof({list(_mkw)}): {type(_e).__name__}: {_e}")

        # If proba failed, fall back to predict_oof which returns labels directly
        if _oof_pred is None:
            _pred_method = getattr(predictor, 'predict_oof', None)
            if _pred_method is not None:
                for _mkw in [{}, {'silent': True}]:
                    try:
                        _oof_pred = _pred_method(**_mkw)
                        logging.info(f"OOF OK via predict_oof({list(_mkw)}): {len(_oof_pred)} rows")
                        break
                    except Exception as _e:
                        logging.warning(f"predict_oof({list(_mkw)}): {type(_e).__name__}: {_e}")

        # When bagging is disabled: use explicit val split (unseen) if available,
        # otherwise fall back to in-sample predictions.
        if _oof_pred is None:
            eval_df = df_val if df_val is not None else df
            source = "validation set" if df_val is not None else "in-sample (no val split provided)"
            logging.info(f"OOF unavailable — generating predictions on {source}.")
            try:
                X_plot = eval_df.drop(columns=[target_column])
                if pt == 'regression':
                    _oof_pred = predictor.predict(X_plot)
                else:
                    _raw = predictor.predict_proba(X_plot)
                    if isinstance(_raw, pd.DataFrame):
                        _oof_pred_proba = _raw
                        _oof_pred = _raw.idxmax(axis=1)
                    else:
                        _oof_pred_proba = _raw
                        pos = predictor.class_labels[1]
                        neg = predictor.class_labels[0]
                        _oof_pred = _raw.map(lambda p: pos if p >= 0.5 else neg)
                _oof_pred.index = eval_df.index
                if _oof_pred_proba is not None:
                    _oof_pred_proba.index = eval_df.index
                # Point df to eval_df so y_test_eval is taken from the right rows
                df = eval_df
                logging.info(f"Predictions generated: {len(_oof_pred)} rows")
            except Exception as _e:
                logging.warning(f"Prediction fallback failed: {_e}")
        # ─────────────────────────────────────────────────────────────────────────

        if stop_event and stop_event.is_set():
            raise UserError("training_interrupted")

        feature_importance_df = predictor.feature_importance(df_train)
        feature_importance_plot = generate_feature_importance_plot(feature_importance_df, save_dir=plot_dir)

        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best
        task_type = predictor.problem_type

        class_labels = predictor.class_labels if hasattr(predictor, 'class_labels') else None
        score_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, 'name') else str(predictor.eval_metric)

        # ── Fetch all OOF scalar metrics via leaderboard(extra_metrics=[...]) ──────
        # AutoGluon 1.3 computes these internally from the same OOF folds used during
        # training. Values are returned in natural sign (positive RMSE, 0–1 accuracy).
        # This is the most reliable approach — no need to call get_oof_pred* at all
        # for scalar metrics.
        if task_type == 'binary':
            _extra_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc',
                              'precision_macro', 'recall_macro']
        elif task_type == 'multiclass':
            _extra_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro',
                              'precision_macro', 'recall_macro', 'roc_auc_ovo_macro']
        else:
            _extra_metrics = ['root_mean_squared_error', 'mean_absolute_error', 'r2']

        try:
            # extra_metrics are only requested when we have an explicit holdout val set.
            # With bagging ON, score_val is already OOF-honest; passing training data to
            # extra_metrics would evaluate in-sample (inflated). Secondary metrics for
            # bagging mode are computed directly from _oof_pred further below instead.
            if df_val is not None:
                _lb_extra = predictor.leaderboard(data=df_val, extra_metrics=_extra_metrics, silent=True)
            else:
                _lb_extra = leaderboard  # no extra columns — fall through to _col_val → None
            _best_row = _lb_extra[_lb_extra['model'] == best_model].iloc[0]
            logging.info(f"leaderboard(extra_metrics) OK. Columns: {list(_lb_extra.columns)}")
        except Exception as _e:
            logging.warning(f"leaderboard(extra_metrics) failed: {type(_e).__name__}: {_e}. Using score_val only.")
            _lb_extra = leaderboard
            _best_row = leaderboard[leaderboard['model'] == best_model].iloc[0]

        def _col_val(row, *names):
            """Return the first column present and finite, else None."""
            for n in names:
                if n in row.index:
                    try:
                        v = float(row[n])
                        if pd.notna(v):
                            return v
                    except Exception:
                        pass
            return None

        # Fallback primary value from score_val with sign fix (used when extra_metrics
        # column is absent or NaN).
        _best_score_raw = float(leaderboard[leaderboard['model'] == best_model]['score_val'].values[0])
        _em = predictor.eval_metric
        _gib = getattr(_em, 'greater_is_better', None)
        if _gib is None:
            _gib = (1 == getattr(_em, 'sign', 1))
        if _gib is None:
            _gib = _best_score_raw >= 0
        _primary_val_fallback = _best_score_raw if _gib else -_best_score_raw

        # OOF predictions — used for plots only (not for scalar metric values).
        oof_ok = False
        y_pred = None
        y_proba = None
        y_test_eval = None

        if _oof_pred is not None:
            y_pred = _oof_pred
            y_test_eval = df[target_column].loc[y_pred.index]
            y_proba = _oof_pred_proba.loc[y_pred.index] if _oof_pred_proba is not None else None
            oof_ok = True
        else:
            logging.warning("OOF predictions not available — plots will be skipped.")

        perf_data = {}
        summary = ""

        if task_type == 'binary':
            if class_labels is None or len(class_labels) < 2:
                raise UserError("error_unexpected_training")
            _acc_raw = _col_val(_best_row, 'accuracy')
            acc = _acc_raw if _acc_raw is not None else _primary_val_fallback
            auc = _col_val(_best_row, 'roc_auc')
            _pos_label = class_labels[1] if class_labels and len(class_labels) > 1 else None
            # If extra_metrics didn't give AUC, compute from OOF probabilities
            if auc is None and oof_ok and y_proba is not None and _pos_label is not None:
                try:
                    auc = float(roc_auc_score(y_test_eval, y_proba[_pos_label]))
                except Exception as _e:
                    logging.warning(f"roc_auc_score fallback failed: {_e}")
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test_eval, y_pred=y_pred,
                                                 class_labels=class_labels, save_dir=plot_dir) if oof_ok else None
                },
            }
            if auc is not None:
                perf_data['roc_auc'] = {
                    'value': auc,
                    'plot': generate_metric_plot('roc_auc', y_true=y_test_eval,
                                                 y_proba=y_proba[_pos_label] if (oof_ok and y_proba is not None and _pos_label is not None) else None,
                                                 class_labels=class_labels, save_dir=plot_dir) if (oof_ok and y_proba is not None and _pos_label is not None) else None
                }
            summary = f"Tâche détectée : classification binaire\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique accuracy : {acc:.4f}\n"
            summary += f"Résultat de la métrique ROC AUC : {auc:.4f}\n" if auc is not None else ""

        elif task_type == 'multiclass':
            _acc_raw = _col_val(_best_row, 'accuracy')
            acc = _acc_raw if _acc_raw is not None else _primary_val_fallback
            f1 = _col_val(_best_row, 'f1_macro')
            # If extra_metrics didn't give F1, compute from OOF predictions
            if f1 is None and oof_ok:
                try:
                    f1 = float(f1_score(y_test_eval, y_pred, average='macro'))
                except Exception as _e:
                    logging.warning(f"f1_score fallback failed: {_e}")
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test_eval, y_pred=y_pred,
                                                 class_labels=class_labels, save_dir=plot_dir) if oof_ok else None
                },
            }
            if f1 is not None:
                perf_data['f1_macro'] = {
                    'value': f1,
                    'plot': generate_metric_plot('f1_macro', y_true=y_test_eval, y_pred=y_pred,
                                                 class_labels=class_labels, save_dir=plot_dir) if oof_ok else None
                }
            summary = f"Tâche détectée : classification multiclasse\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique accuracy : {acc:.4f}\n"
            summary += f"Résultat de la métrique F1 macro : {f1:.4f}\n" if f1 is not None else ""

        elif task_type == 'regression':
            rmse = _col_val(_best_row, 'root_mean_squared_error', 'rmse')
            # AutoGluon stores error metrics negated in all leaderboard columns
            if rmse is not None and rmse < 0:
                rmse = -rmse
            r2 = _col_val(_best_row, 'r2')
            # Fallbacks: compute from OOF predictions or use score_val
            if rmse is None:
                if oof_ok:
                    try:
                        rmse = float(mean_squared_error(y_test_eval, y_pred) ** 0.5)
                    except Exception as _e:
                        logging.warning(f"rmse fallback failed: {_e}")
                if rmse is None:
                    rmse = _primary_val_fallback  # always has a value
            if r2 is None and oof_ok:
                try:
                    r2 = float(r2_score(y_test_eval, y_pred))
                except Exception as _e:
                    logging.warning(f"r2_score fallback failed: {_e}")
            if r2 is None and score_metric in ('r2', 'r2_score'):
                r2 = _primary_val_fallback
            perf_data = {
                'rmse': {
                    'value': rmse,
                    'plot': generate_metric_plot('rmse', y_true=y_test_eval, y_pred=y_pred,
                                                 save_dir=plot_dir) if oof_ok else None,
                    'plot_hist': generate_rmse_error_histogram_base64(y_true=y_test_eval, y_pred=y_pred,
                                                                       save_dir=plot_dir) if oof_ok else None
                }
            }
            if r2 is not None:
                perf_data['r2'] = {'value': r2}
            summary = f"Tâche détectée : regression\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique R2 : {r2:.4f}\n" if r2 is not None else ""
            summary += f"Résultat de la métrique RMSE : {rmse:.4f}" if rmse is not None else ""

        results = {
            'best_model': best_model,
            'train_time': float(leaderboard['fit_time'].sum()),
            'task_type': task_type,
            'metrics': perf_data,
            'feature_importance_plot': feature_importance_plot,
            'feature_importance': {str(k): float(v) for k, v in feature_importance_df['importance'].items() if np.isfinite(v)},
            'model_path': save_path,
            'leaderboard': leaderboard.to_dict(orient='records'),
            'summary_LLM' : summary
        }

        return results

    except UserError:
        raise
    except Exception:
        logging.exception("Unexpected error during training")
        raise UserError("error_unexpected_training")


def train_model_timeseries(df, target_column, time_column, time_limit, save_path, stop_event, plot_dir,
                           prediction_length_percent=None, eval_metric=None, excluded_model_types=None):
    # Ensure datetime
    try:
        df[time_column] = pd.to_datetime(df[time_column])
    except Exception:
        raise UserError("error_invalid_time_column")

    df = df.sort_values(time_column)
    df["item_id"] = "series_1"  # single series
    df = df.rename(columns={target_column: "target"})

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column=time_column
    )

    try:
        percent = float(prediction_length_percent) if prediction_length_percent is not None else 10.0
    except Exception:
        percent = 10.0
    percent = max(5.0, min(30.0, percent))
    prediction_length = max(1, int(len(ts_df) * (percent / 100.0)))

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="target",
        path=save_path,
        eval_metric=eval_metric,
    )

    if stop_event and stop_event.is_set():
        raise UserError("training_interrupted")

    fit_start = datetime.now()
    ts_fit_kwargs = {'time_limit': time_limit, 'presets': 'medium_quality'}
    if excluded_model_types:
        ts_fit_kwargs['excluded_model_types'] = excluded_model_types
    predictor.fit(train_data=ts_df, **ts_fit_kwargs)
    fit_elapsed = (datetime.now() - fit_start).total_seconds()

    if stop_event and stop_event.is_set():
        raise UserError("training_interrupted")

    forecasts = predictor.predict(ts_df)
    test_slice = ts_df.slice_by_timestep(-prediction_length, None)

    # forecasts columns usually contain quantiles; use mean if available, else first column
    if "mean" in forecasts.columns:
        forecast_series = forecasts["mean"]
    else:
        forecast_series = forecasts.iloc[:, 0]

    y_true = test_slice["target"].to_numpy()
    y_pred = forecast_series.to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)

    # Plot using plain columns
    df_sorted = df.sort_values(time_column)
    actual_time = df_sorted[time_column].reset_index(drop=True)
    actual_values = df_sorted["target"].reset_index(drop=True)
    forecast_time = forecasts.index.get_level_values("timestamp")
    forecast_values = forecast_series.to_numpy()

    total_len = len(actual_time)
    train_end_idx = max(0, total_len - prediction_length - 1)
    val_start_time = actual_time.iloc[total_len - prediction_length] if total_len > prediction_length else actual_time.iloc[0]
    val_end_time = actual_time.iloc[-1]
    forecast_start = forecast_time[0] if len(forecast_time) > 0 else val_end_time
    forecast_end = forecast_time[-1] if len(forecast_time) > 0 else val_end_time

    fig, ax = plt.subplots(figsize=(10, 4))

    # Background region shading
    ax.axvspan(actual_time.iloc[0], actual_time.iloc[train_end_idx],
               alpha=0.08, color='steelblue', label='_train_region')
    ax.axvspan(val_start_time, val_end_time,
               alpha=0.12, color='orange', label='_val_region')
    ax.axvspan(forecast_start, forecast_end,
               alpha=0.12, color='green', label='_forecast_region')

    # Dummy patches for region legend entries
    train_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='Train')
    val_patch = mpatches.Patch(color='orange', alpha=0.4, label='Validation')
    fc_patch = mpatches.Patch(color='green', alpha=0.4, label='Forecast region')

    ax.plot(forecast_time, forecast_values, color='tomato', linewidth=1.5, linestyle='--', label='Forecast', zorder=2)
    ax.plot(actual_time, actual_values, color='steelblue', linewidth=1.5, label='Actual', zorder=3)

    ax.set_xlabel(time_column, fontsize=10)
    ax.set_ylabel(target_column, fontsize=10)
    ax.set_title("Forecast vs Actual", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    # Remove the dummy axvspan entries (prefixed with _)
    handles = [h for h, l in zip(handles, labels) if not l.startswith('_')]
    labels = [l for l in labels if not l.startswith('_')]
    handles += [train_patch, val_patch, fc_patch]
    labels += ['Train', 'Validation', 'Forecast region']
    ax.legend(handles, labels, fontsize=8, loc='best')

    plt.tight_layout()
    save_plot_to_disk(plt, plot_dir, "forecast_vs_actual.png")
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120)
    buffer.seek(0)
    forecast_plot_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    best_model_name = None
    leaderboard_records = None
    try:
        lb = predictor.leaderboard(silent=True)
        if not lb.empty and 'model' in lb.columns:
            best_model_name = lb.iloc[0]['model']
            # Rename columns to match tabular leaderboard format expected by the frontend
            lb_out = lb[['model']].copy()
            score_col = next((c for c in ['score_val', 'score', 'val_score'] if c in lb.columns), None)
            fit_col   = next((c for c in ['fit_time', 'fit_time_marginal'] if c in lb.columns), None)
            pred_col  = next((c for c in ['pred_time_val', 'predict_time', 'inference_time'] if c in lb.columns), None)
            lb_out['score_val']     = lb[score_col].round(4) if score_col else None
            lb_out['fit_time']      = lb[fit_col].round(2)   if fit_col   else None
            lb_out['pred_time_val'] = lb[pred_col].round(4)  if pred_col  else None
            leaderboard_records = lb_out.to_dict(orient='records')
    except Exception as _e:
        logging.warning(f"Time series leaderboard failed: {_e}")

    results = {
        "best_model": best_model_name,
        "train_time": fit_elapsed,
        "task_type": "timeseries",
        "metrics": {
            "mae": {"value": mae},
            "rmse": {"value": rmse},
            "mape": {"value": mape},
        },
        "feature_importance_plot": None,
        "model_path": save_path,
        "leaderboard": leaderboard_records,
        "summary_LLM": None,
        "forecast_plot": forecast_plot_b64,
        "prediction_length": prediction_length,
    }
    return results


# Tree-based model class name keywords that support shap.TreeExplainer
_TREE_MODEL_KEYWORDS = (
    'ExtraTrees', 'RandomForest', 'XGBoost', 'XGB',
    'LightGBM', 'LGBM', 'CatBoost', 'GradientBoosting', 'DecisionTree',
)

def _unwrap_ag_model(ag_model):
    """Extract the raw sklearn/xgb/lgb estimator from an AutoGluon model wrapper.
    Returns None if the structure is not recognised."""
    try:
        raw = ag_model
        # BaggedEnsembleModel stores fold models in .models
        if hasattr(raw, 'models') and raw.models:
            raw = raw.models[0]
        # Most AutoGluon wrappers store the actual estimator in .model
        if hasattr(raw, 'model'):
            return raw.model
        # Return raw itself as a last attempt (may already be an sklearn estimator)
        return raw
    except Exception:
        pass
    return None


def _build_shap_explainer(predictor, X):
    """Return the fastest SHAP explainer available for this predictor.

    Prefers shap.TreeExplainer (milliseconds) when the best base model is a
    tree ensemble; falls back to shap.Explainer (PermutationExplainer, slow)
    otherwise.
    """
    trainer = predictor._trainer

    # Find best non-ensemble model (WeightedEnsemble wraps others, not a tree itself)
    try:
        lb = trainer.leaderboard(silent=True)
        non_ens = lb[~lb['model'].str.contains('WeightedEnsemble', na=False)]
        best_base_name = non_ens.iloc[0]['model']
        base_model = trainer.load_model(best_base_name)
    except Exception:
        base_model = trainer.load_model(trainer.model_best)

    raw = _unwrap_ag_model(base_model)
    if raw is not None and any(k in type(raw).__name__ for k in _TREE_MODEL_KEYWORDS):
        try:
            explainer = shap.TreeExplainer(raw)
            explainer(X.iloc[:1])          # quick smoke-test
            logging.info(f"SHAP: using TreeExplainer with {type(raw).__name__}")
            return explainer
        except Exception as exc:
            logging.warning(f"TreeExplainer failed ({exc}), falling back to PermutationExplainer")

    # Universal fallback — slow but works with any model
    best_model = trainer.load_model(trainer.model_best)
    if predictor.problem_type in ('binary', 'multiclass'):
        fn = best_model.predict_proba
    else:
        fn = best_model.predict
    logging.info("SHAP: using PermutationExplainer (generic fallback)")
    return shap.Explainer(fn, X)


# Generate SHAP summary plot for model explanations
def generate_shap_plot(model_path, df, target_column):
    try:
        predictor = TabularPredictor.load(model_path)

        if target_column not in df.columns:
            raise UserError("error_missing_target_column")

        X = df.drop(columns=[target_column])
        X = X.sample(n=min(100, len(X)))

        if model_path not in _shap_explainer_cache:
            _shap_explainer_cache[model_path] = _build_shap_explainer(predictor, X)
        explainer = _shap_explainer_cache[model_path]

        shap_values = explainer(X)
        shap_summary_plot = shap_plot_to_base64(shap.summary_plot, shap_values, X)

        return {'shap_summary_plot': shap_summary_plot}

    except UserError:
        raise
    except Exception:
        raise UserError("error_shap")

# Plot distribution of predicted values for regression
def plot_regression_distribution(y_pred):
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred, bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Frequency")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

# Plot boxplot to detect outliers in predictions
def plot_prediction_outliers(y_pred):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=y_pred, color="tomato")
    plt.title("Outlier Detection in Predictions")
    plt.xlabel("Predicted Value")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

# Plot distribution of predicted classes for classification
def plot_predicted_class_distribution(y_pred):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred, palette='pastel')
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Occurrences")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

# Plot predicted probability distributions for each class
def plot_prediction_confidence(y_proba_df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=y_proba_df, orient='h', palette='Set2')
    plt.title("Predicted Probability Distribution by Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Class")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

# Predict using a trained model and generate relevant plots
def predict_model(csv_path, model_path):
    try:
        predictor = TabularPredictor.load(model_path)

        df = load_dataframe(csv_path)

        # Remove target column if present
        if predictor.label in df.columns:
            df = df.drop(columns=[predictor.label])

        expected_cols = predictor.feature_metadata.get_features()
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)

        if missing_cols:
            raise UserError("error_missing_columns")

        if extra_cols:
            raise UserError("error_extra_columns")

        predictions = predictor.predict(df)
        task_type = predictor.problem_type
        output_data = {
            'predictions': predictions.reset_index().to_dict(orient='records'),
            'plots': {}
        }

        # Generate plots based on task type
        if task_type == 'regression':
            output_data['plots'] = {
                'distribution': plot_regression_distribution(predictions),
            }

        elif task_type == 'binary':
            y_proba = predictor.predict_proba(df)
            output_data['plots'] = {
                'class_distribution': plot_predicted_class_distribution(predictions),
                'confidence': plot_prediction_confidence(y_proba)
            }

        elif task_type == 'multiclass':
            y_proba = predictor.predict_proba(df)
            output_data['plots'] = {
                'class_distribution': plot_predicted_class_distribution(predictions),
                'confidence': plot_prediction_confidence(y_proba)
            }

        return output_data

    except UserError:
        raise
    except Exception:
        raise UserError("error_unexpected_prediction")


