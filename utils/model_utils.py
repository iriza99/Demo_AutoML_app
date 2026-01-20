from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
import pandas as pd
import matplotlib
import logging
import traceback

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np
import shap
from datetime import datetime
import io


from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

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
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances['importance'], y=importances.index, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
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

# Train an AutoGluon model and return training results and plots
def train_model(df, target_column, time_limit, save_path, stop_event=None, time_column=None, prediction_length_percent=None):
    try:
        save_path = os.path.abspath(save_path)

        plot_dir = os.path.join(save_path, "plot_train_results")
        os.makedirs(plot_dir, exist_ok=True)

        if df[target_column].isnull().any():
            raise UserError("error_missing_target_values")

        # Time series branch
        if time_column:
            return train_model_timeseries(
                df,
                target_column,
                time_column,
                time_limit,
                save_path,
                stop_event,
                plot_dir,
                prediction_length_percent
            )

        predictor = TabularPredictor(label=target_column, path=save_path)

        if stop_event and stop_event.is_set():
            raise UserError("training_interrupted")
        
        predictor.fit(df, time_limit=time_limit)

        if stop_event and stop_event.is_set():
            raise UserError("training_interrupted")

        feature_importance_df = predictor.feature_importance(df)
        feature_importance_plot = generate_feature_importance_plot(feature_importance_df, save_dir=plot_dir)

        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best
        task_type = predictor.problem_type

        y_test = df[target_column]
        y_pred = predictor.predict(df)
        y_proba = predictor.predict_proba(df) if task_type == 'binary' else None
        class_labels = predictor.class_labels if hasattr(predictor, 'class_labels') else None

        perf_data = {}
        summary = ""

        if task_type == 'binary':
            # Binary classification metrics and plots
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba[class_labels[1]])
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test, y_pred=y_pred, class_labels=class_labels, save_dir = plot_dir)
                },
                'roc_auc': {
                    'value': auc,
                    'plot': generate_metric_plot('roc_auc', y_true=y_test, y_proba=y_proba[class_labels[1]], class_labels=class_labels, save_dir = plot_dir)
                }
            }
            summary = f"Tâche détectée : classification binaire\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique accuracy : {perf_data['accuracy']['value']:.4f}\n"
            summary += f"Résultat de la métrique ROC AUC : {perf_data['roc_auc']['value']:.4f}\n"

        elif task_type == 'multiclass':
            # Multiclass classification metrics and plots
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test, y_pred=y_pred, class_labels=class_labels, save_dir = plot_dir)
                },
                'f1_macro': {
                    'value': f1,
                    'plot': generate_metric_plot('f1_macro', y_true=y_test, y_pred=y_pred, class_labels=class_labels, save_dir = plot_dir)
                }
            }
            summary = f"Tâche détectée : classification multiclasse\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique accuracy : {perf_data['accuracy']['value']:.4f}\n"
            summary += f"Résultat de la métrique F1 macro : {perf_data['f1_macro']['value']:.4f}\n"

        elif task_type == 'regression':
            # Regression metrics and plots
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            perf_data = {
                'r2': {
                    'value': r2
                },
                'rmse': {
                    'value': rmse,
                    'plot': generate_metric_plot('rmse', y_true=y_test, y_pred=y_pred, save_dir = plot_dir),
                    'plot_hist': generate_rmse_error_histogram_base64(y_true=y_test, y_pred=y_pred, save_dir = plot_dir)
                }
            }
            summary = f"Tâche détectée : regression\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += f"Résultat de la métrique R2 : {perf_data['r2']['value']:.4f}\n"
            summary += f"Résultat de la métrique RMSE : {perf_data['rmse']['value']:.4f}"

        score_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, 'name') else str(predictor.eval_metric)

        results = {
            'best_model': best_model,
            'train_time': float(leaderboard['fit_time'].sum()),
            'task_type': task_type,
            'metrics': perf_data,
            'feature_importance_plot': feature_importance_plot,
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


def train_model_timeseries(df, target_column, time_column, time_limit, save_path, stop_event, plot_dir, prediction_length_percent=None):
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
    )

    if stop_event and stop_event.is_set():
        raise UserError("training_interrupted")

    predictor.fit(train_data=ts_df, time_limit=time_limit, presets="medium_quality")

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
    actual_time = df_sorted[time_column]
    actual_values = df_sorted["target"]
    forecast_time = forecasts.index.get_level_values("timestamp")
    forecast_values = forecast_series.to_numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(actual_time, actual_values, label="Actual")
    plt.plot(forecast_time, forecast_values, label="Forecast")
    plt.legend()
    plt.title("Forecast vs Actual")
    plt.tight_layout()
    save_plot_to_disk(plt, plot_dir, "forecast_vs_actual.png")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    forecast_plot_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    best_model_name = None
    try:
        lb = predictor.leaderboard(silent=True)
        if not lb.empty and 'model' in lb.columns:
            best_model_name = lb.iloc[0]['model']
    except Exception:
        best_model_name = None

    results = {
        "best_model": best_model_name,
        "train_time": float(time_limit),
        "task_type": "timeseries",
        "metrics": {
            "mae": {"value": mae},
            "rmse": {"value": rmse},
            "mape": {"value": mape},
        },
        "feature_importance_plot": None,
        "model_path": save_path,
        "leaderboard": None,
        "summary_LLM": None,
        "forecast_plot": forecast_plot_b64,
        "prediction_length": prediction_length,
    }
    return results


# Generate SHAP summary plot for model explanations
def generate_shap_plot(model_path, df, target_column):
    try:
        predictor = TabularPredictor.load(model_path)
        model = predictor._trainer.load_model(predictor._trainer.model_best)

        if target_column not in df.columns:
            raise UserError("error_missing_target_column")

        X = df.drop(columns=[target_column])

        X = X.sample(n=min(100, len(X)), random_state=42)  

        # SHAP explainer for classification or regression
        if predictor.problem_type in ["binary", "multiclass"]:
            explainer = shap.Explainer(model.predict_proba, X)
        else:
            explainer = shap.Explainer(model.predict, X)

        shap_values = explainer(X)
        shap_summary_plot = shap_plot_to_base64(shap.summary_plot, shap_values, X)

        return {
            'shap_summary_plot': shap_summary_plot,
        }

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

        ext = os.path.splitext(csv_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(csv_path)
        elif ext in ['.xls', '.xlsx', '.xlsm']:
            df = pd.read_excel(csv_path)
        elif ext == '.arff':
            import arff
            with open(csv_path, 'r') as f:
                arff_data = arff.load(f)
            df = pd.DataFrame(arff_data['data'], columns=[a[0] for a in arff_data['attributes']])
        else:
            raise UserError("error_unsupported_file_format")

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


