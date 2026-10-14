"""PDF report of a training run.

Built only from the data the app already has (results, plot data, column
statistics), so it is fast, works without an assistant key and never invents
numbers. Figures come from utils.plots, in the same style as the app.
"""
import base64
import io
import logging
import math
import os
from datetime import datetime

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from xml.sax.saxutils import escape

from utils import plots

INK = colors.HexColor(plots.INK)
MUTED = colors.HexColor(plots.MUTED)
GRID = colors.HexColor(plots.GRID)
PRIMARY = colors.HexColor(plots.PRIMARY)
SOFT = colors.HexColor("#f3f6fd")

TEXT = {
    "en": {
        "title": "Model training report", "generated": "Generated on {date}",
        "overview": "Overview", "dataset": "Dataset", "rows": "Records", "columns": "Columns",
        "target": "Variable to predict", "task": "Task", "best_model": "Best model",
        "train_time": "Training time", "gpu": "GPU used", "yes": "Yes", "no": "No",
        "excluded": "Excluded variables", "none": "None",
        "metrics": "Performance", "metrics_note": "Measured on data the model did not see while training "
                                                  "(cross-validation or holdout).",
        "metric": "Metric", "value": "Value", "meaning": "What it means",
        "figures": "Figures", "importance": "Variable importance", "shap": "SHAP explanation",
        "forecast": "Forecast",
        "models": "Models compared", "model": "Model", "score": "Validation score", "fit": "Training (s)",
        "data": "Variables in the dataset", "name": "Name", "type": "Type", "missing": "Missing",
        "unique": "Distinct", "summary": "Summary",
        "binary": "Binary classification", "multiclass": "Multiclass classification",
        "regression": "Regression", "timeseries": "Time series forecast",
        "Numeric": "Numeric", "Categorical": "Categorical", "Date": "Date",
        "seconds": "{s} s", "page": "Page {n}",
        "imputation": "Missing values", "imputation_none": "Handled automatically by AutoML",
        "m_roc_auc": "Probability that a random positive case gets a higher score than a random negative one (0.5 = chance, 1 = perfect).",
        "m_accuracy": "Share of correct predictions. Misleading when one class is much more frequent.",
        "m_f1_macro": "Average F1 over classes; each class counts the same, however rare.",
        "m_r2": "Share of the variability of the target explained by the model (1 = perfect, 0 = predicting the mean).",
        "m_rmse": "Typical size of the error, in the units of the target. Lower is better.",
        "m_mae": "Average absolute error, in the units of the target. Lower is better.",
        "m_mape": "Average error as a percentage of the actual value. Lower is better.",
    },
    "es": {
        "title": "Informe de entrenamiento del modelo", "generated": "Generado el {date}",
        "overview": "Resumen", "dataset": "Conjunto de datos", "rows": "Registros", "columns": "Columnas",
        "target": "Variable a predecir", "task": "Tarea", "best_model": "Mejor modelo",
        "train_time": "Tiempo de entrenamiento", "gpu": "GPU utilizada", "yes": "Sí", "no": "No",
        "excluded": "Variables excluidas", "none": "Ninguna",
        "metrics": "Rendimiento", "metrics_note": "Medido sobre datos que el modelo no vio al entrenar "
                                                   "(validación cruzada o conjunto reservado).",
        "metric": "Métrica", "value": "Valor", "meaning": "Qué significa",
        "figures": "Figuras", "importance": "Importancia de las variables", "shap": "Explicación SHAP",
        "forecast": "Previsión",
        "models": "Modelos comparados", "model": "Modelo", "score": "Puntuación de validación", "fit": "Entrenamiento (s)",
        "data": "Variables del conjunto de datos", "name": "Nombre", "type": "Tipo", "missing": "Faltantes",
        "unique": "Distintos", "summary": "Resumen",
        "binary": "Clasificación binaria", "multiclass": "Clasificación multiclase",
        "regression": "Regresión", "timeseries": "Previsión de series temporales",
        "Numeric": "Numérica", "Categorical": "Categórica", "Date": "Fecha",
        "seconds": "{s} s", "page": "Página {n}",
        "imputation": "Valores faltantes", "imputation_none": "Gestionados automáticamente por AutoML",
        "m_roc_auc": "Probabilidad de que un caso positivo al azar reciba una puntuación mayor que uno negativo (0,5 = azar, 1 = perfecto).",
        "m_accuracy": "Proporción de predicciones correctas. Engañosa si una clase es mucho más frecuente.",
        "m_f1_macro": "F1 medio entre clases; todas cuentan igual, aunque sean raras.",
        "m_r2": "Proporción de la variabilidad de la variable explicada por el modelo (1 = perfecto, 0 = predecir la media).",
        "m_rmse": "Tamaño típico del error, en las unidades de la variable. Menor es mejor.",
        "m_mae": "Error absoluto medio, en las unidades de la variable. Menor es mejor.",
        "m_mape": "Error medio en porcentaje del valor real. Menor es mejor.",
    },
    "fr": {
        "title": "Rapport d'entraînement du modèle", "generated": "Généré le {date}",
        "overview": "Résumé", "dataset": "Jeu de données", "rows": "Enregistrements", "columns": "Colonnes",
        "target": "Variable à prédire", "task": "Tâche", "best_model": "Meilleur modèle",
        "train_time": "Temps d'entraînement", "gpu": "GPU utilisé", "yes": "Oui", "no": "Non",
        "excluded": "Variables exclues", "none": "Aucune",
        "metrics": "Performance", "metrics_note": "Mesurée sur des données non vues à l'entraînement "
                                                  "(validation croisée ou jeu réservé).",
        "metric": "Métrique", "value": "Valeur", "meaning": "Signification",
        "figures": "Figures", "importance": "Importance des variables", "shap": "Explication SHAP",
        "forecast": "Prévision",
        "models": "Modèles comparés", "model": "Modèle", "score": "Score de validation", "fit": "Entraînement (s)",
        "data": "Variables du jeu de données", "name": "Nom", "type": "Type", "missing": "Manquants",
        "unique": "Distincts", "summary": "Résumé",
        "binary": "Classification binaire", "multiclass": "Classification multiclasse",
        "regression": "Régression", "timeseries": "Prévision de séries temporelles",
        "Numeric": "Numérique", "Categorical": "Catégorielle", "Date": "Date",
        "seconds": "{s} s", "page": "Page {n}",
        "imputation": "Valeurs manquantes", "imputation_none": "Gérées automatiquement par AutoML",
        "m_roc_auc": "Probabilité qu'un cas positif au hasard obtienne un score plus élevé qu'un cas négatif (0,5 = hasard, 1 = parfait).",
        "m_accuracy": "Part de prédictions correctes. Trompeuse si une classe est beaucoup plus fréquente.",
        "m_f1_macro": "F1 moyen entre classes ; chaque classe compte autant, même rare.",
        "m_r2": "Part de la variabilité de la cible expliquée par le modèle (1 = parfait, 0 = prédire la moyenne).",
        "m_rmse": "Taille typique de l'erreur, dans les unités de la cible. Plus bas est mieux.",
        "m_mae": "Erreur absolue moyenne, dans les unités de la cible. Plus bas est mieux.",
        "m_mape": "Erreur moyenne en pourcentage de la valeur réelle. Plus bas est mieux.",
    },
}

METRIC_LABELS = {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "f1_macro": "F1 macro", "r2": "R²",
                 "rmse": "RMSE", "mae": "MAE", "mape": "MAPE"}
METRIC_ORDER = list(METRIC_LABELS)

_FONTS = None


def _fonts():
    """DejaVu (shipped with matplotlib) covers accents and symbols like − and ²."""
    global _FONTS
    if _FONTS is None:
        try:
            base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
            pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
            _FONTS = ("DejaVu", "DejaVu-Bold")
        except Exception:
            logging.exception("DejaVu font not available, falling back to Helvetica")
            _FONTS = ("Helvetica", "Helvetica-Bold")
    return _FONTS


def _styles():
    regular, bold = _fonts()
    return {
        "title": ParagraphStyle("title", fontName=bold, fontSize=20, leading=24, textColor=INK, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontName=regular, fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13, leading=17, textColor=INK, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=regular, fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT),
        "muted": ParagraphStyle("muted", fontName=regular, fontSize=8.5, leading=12, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName=regular, fontSize=8.5, leading=11, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName=bold, fontSize=8.5, leading=11, textColor=INK),
        "head": ParagraphStyle("head", fontName=bold, fontSize=8.5, leading=11, textColor=MUTED),
        "kpi": ParagraphStyle("kpi", fontName=bold, fontSize=17, leading=20, textColor=INK),
        "kpil": ParagraphStyle("kpil", fontName=regular, fontSize=8.5, leading=11, textColor=MUTED),
    }


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        if abs(value) >= 1e6 or (0 < abs(value) < 0.001):
            return f"{value:.{digits}g}"
        if abs(value) >= 1000:
            return f"{value:,.1f}".rstrip("0").rstrip(".")
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _table(rows, widths, header=True, zebra=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID),
    ]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    table.setStyle(TableStyle(style))
    return table


def _image(b64, max_width, max_height=95 * mm):
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        w, h = ImageReader(io.BytesIO(raw)).getSize()
        scale = min(max_width / w, max_height / h)
        return Image(io.BytesIO(raw), width=w * scale, height=h * scale)
    except Exception:
        logging.exception("Could not embed an image in the report")
        return None


def _side_by_side(images, width):
    images = [i for i in images if i is not None]
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    table = Table([images[:2]], colWidths=[width / 2] * 2, hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return table


def _figures(data):
    """PNG figures (base64) for the results, regenerated in the app's style."""
    plot_data = data.get("plot_data") or {}
    figures = []
    try:
        if "confusion" in plot_data:
            figures.append(plots.confusion_matrix_figure(plot_data["confusion"], plot_data["labels"]))
            roc = plot_data.get("roc")
            if roc:
                figures.append(plots.roc_figure(roc["fpr"], roc["tpr"], roc.get("auc")))
        elif "y_true" in plot_data:
            rmse = (data.get("metrics") or {}).get("rmse")
            figures.append(plots.scatter_figure(plot_data["y_true"], plot_data["y_pred"], rmse))
            figures.append(plots.error_histogram_figure(plot_data["y_true"], plot_data["y_pred"]))
    except Exception:
        logging.exception("Could not draw result figures for the report")
    return figures


def _column_summary(stat, T):
    if stat.get("type") == "Numeric" and stat.get("mean") is not None:
        return f"{_fmt(stat.get('mean'), 3)} ± {_fmt(stat.get('std'), 3)}  [{_fmt(stat.get('min'), 3)} – {_fmt(stat.get('max'), 3)}]"
    top = stat.get("top") or []
    parts = []
    for item in top[:3]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            parts.append(f"{item[0]} ({item[1]})")
    return ", ".join(parts)


def _imputation_text(column, choice):
    if isinstance(choice, dict):
        method = choice.get("method") or ""
        if choice.get("constant") not in (None, ""):
            method += f" = {choice['constant']}"
        return f"{column}: {method}"
    return f"{column}: {choice}"


def build_report(data, lang="en"):
    """Return a BytesIO with the PDF. `data` is the JSON the app sends to /download_pdf."""
    lang = lang if lang in TEXT else "en"
    T = TEXT[lang]
    plots.use_language(lang)
    S = _styles()
    buffer = io.BytesIO()
    margin = 18 * mm
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=T["title"], author="AutoML assistant")
    width = A4[0] - 2 * margin
    regular, _ = _fonts()

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 9 * mm, str(data.get("dataset_name") or ""))
        canvas.drawRightString(A4[0] - margin, 9 * mm, T["page"].format(n=_doc.page))
        canvas.restoreState()

    task = data.get("task_type")
    metrics = {k: v for k, v in (data.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and math.isfinite(v)}
    story = [
        _p(T["title"], S["title"]),
        _p(T["generated"].format(date=datetime.now().strftime("%Y-%m-%d %H:%M")), S["subtitle"]),
    ]

    # KPI strip
    kpis = [k for k in METRIC_ORDER if k in metrics][:4]
    if kpis:
        cells = [[_p(_fmt(metrics[k]), S["kpi"]), _p(METRIC_LABELS[k], S["kpil"])] for k in kpis]
        strip = Table([[c for c in cells]], colWidths=[width / len(kpis)] * len(kpis), hAlign="LEFT")
        strip.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SOFT),
            ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story += [strip, Spacer(1, 4)]

    # Overview
    excluded = [str(c) for c in (data.get("excluded") or [])]
    imputation = data.get("imputation") or {}
    train_time = data.get("train_time")
    overview = [
        (T["dataset"], data.get("dataset_name") or "—"),
        (T["rows"], f"{data['n_rows']:,}" if isinstance(data.get("n_rows"), int) else "—"),
        (T["columns"], f"{data['n_columns']:,}" if isinstance(data.get("n_columns"), int) else "—"),
        (T["target"], data.get("target_column") or "—"),
        (T["task"], T.get(task, task or "—")),
        (T["best_model"], data.get("best_model") or "—"),
        (T["train_time"], T["seconds"].format(s=_fmt(float(train_time), 3)) if isinstance(train_time, (int, float)) else "—"),
        (T["gpu"], T["yes"] if data.get("used_gpu") else T["no"]),
        (T["excluded"], ", ".join(excluded) if excluded else T["none"]),
        (T["imputation"], ", ".join(_imputation_text(k, v) for k, v in imputation.items()) if imputation else T["imputation_none"]),
    ]
    story.append(_p(T["overview"], S["h2"]))
    story.append(_table([[_p(k, S["head"]), _p(v, S["cell"])] for k, v in overview],
                        [width * 0.3, width * 0.7], header=False, zebra=False))

    # Metrics
    if metrics:
        rows = [[_p(T["metric"], S["head"]), _p(T["value"], S["head"]), _p(T["meaning"], S["head"])]]
        for key in [k for k in METRIC_ORDER if k in metrics] + [k for k in metrics if k not in METRIC_ORDER]:
            rows.append([_p(METRIC_LABELS.get(key, key), S["cellb"]), _p(_fmt(metrics[key]), S["cell"]),
                         _p(T.get(f"m_{key}", ""), S["cell"])])
        story += [_p(T["metrics"], S["h2"]), _p(T["metrics_note"], S["muted"]), Spacer(1, 4),
                  _table(rows, [width * 0.18, width * 0.14, width * 0.68])]

    # Figures
    result_figures = _figures(data)
    if data.get("forecast_plot"):
        result_figures.append(data["forecast_plot"])
    if result_figures:
        story.append(_p(T["figures"], S["h2"]))
        half = width / 2 - 6
        for i in range(0, len(result_figures), 2):
            pair = [_image(b, half if len(result_figures) > 1 else width, 80 * mm) for b in result_figures[i:i + 2]]
            block = _side_by_side(pair, width)
            if block is not None:
                story += [block, Spacer(1, 6)]

    importance = data.get("feature_importance") or {}
    importance_png = None
    if importance:
        try:
            names, values = zip(*[(k, float(v)) for k, v in importance.items()])
            importance_png = plots.importance_figure(list(names), list(values))
        except Exception:
            logging.exception("Could not draw importance for the report")
    importance_png = importance_png or data.get("feature_importance_plot")
    if importance_png:
        story.append(KeepTogether([_p(T["importance"], S["h2"]), _image(importance_png, width * 0.85, 110 * mm)]))

    if data.get("shap_plot"):
        img = _image(data["shap_plot"], width, 120 * mm)
        if img is not None:
            story.append(KeepTogether([_p(T["shap"], S["h2"]), img]))

    # Leaderboard
    leaderboard = [r for r in (data.get("leaderboard") or []) if isinstance(r, dict) and r.get("model")]
    if leaderboard:
        rows = [[_p(T["model"], S["head"]), _p(T["score"], S["head"]), _p(T["fit"], S["head"])]]
        for r in leaderboard[:12]:
            score, fit = r.get("score_val"), r.get("fit_time")
            rows.append([_p(r["model"], S["cell"]),
                         _p(_fmt(float(score)) if isinstance(score, (int, float)) else "—", S["cell"]),
                         _p(_fmt(float(fit), 3) if isinstance(fit, (int, float)) else "—", S["cell"])])
        story += [_p(T["models"], S["h2"]), _table(rows, [width * 0.5, width * 0.25, width * 0.25])]

    # Columns
    stats = [s for s in (data.get("stats") or []) if isinstance(s, dict) and s.get("name") is not None]
    if stats:
        rows = [[_p(T[h], S["head"]) for h in ("name", "type", "missing", "unique", "summary")]]
        for s in stats[:200]:
            name = str(s["name"])
            style = S["cellb"] if name == data.get("target_column") else S["cell"]
            rows.append([_p(name, style), _p(T.get(s.get("type"), s.get("type") or ""), S["cell"]),
                         _p(s.get("missing", ""), S["cell"]), _p(s.get("unique", ""), S["cell"]),
                         _p(_column_summary(s, T), S["cell"])])
        story += [_p(T["data"], S["h2"]),
                  _table(rows, [width * 0.25, width * 0.13, width * 0.12, width * 0.12, width * 0.38])]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer
