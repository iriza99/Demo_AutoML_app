"""All static figures (PNG) in one visual style, plus the raw data behind them.

The app draws the main charts natively in the browser from `plot_data`; these
PNGs are used for the PDF report, the plot downloads, the prediction view and
as a fallback. Both share the app's palette (validated for colour-blind
separation) so a figure looks the same wherever it appears.
"""
import base64
import contextvars
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Theme ─────────────────────────────────────────────────────────────────────
INK = "#172033"
MUTED = "#5f6b80"
GRID = "#e2e7f0"
SURFACE = "#ffffff"
SERIES = ["#3355d1", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
PRIMARY = SERIES[0]
# Sequential ramp for the confusion matrix (one hue, light -> dark)
SEQ = ["#f3f6fd", "#d7e0f9", "#aebff2", "#7f98e6", "#5274db", "#3355d1", "#253f9e"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})

# ── Language ──────────────────────────────────────────────────────────────────
_LANG = contextvars.ContextVar("plot_lang", default="en")

LABELS = {
    "en": {
        "cm_title": "Confusion matrix", "actual": "Actual", "predicted": "Predicted",
        "roc_title": "ROC curve", "tpr": "Sensitivity (true positive rate)", "fpr": "1 − specificity (false positive rate)",
        "chance": "Chance", "model": "Model",
        "f1_title": "F1 score by class", "f1": "F1 score",
        "scatter_title": "Predicted vs actual", "error_title": "Prediction errors",
        "error": "Error (predicted − actual)", "count": "Records", "perfect": "Perfect prediction",
        "importance_title": "Variable importance", "importance": "Importance (drop in score when shuffled)",
        "pred_dist_title": "Distribution of predictions", "pred_value": "Predicted value",
        "pred_box_title": "Spread of predictions",
        "class_dist_title": "Predicted classes", "class": "Class",
        "confidence_title": "Predicted probability by class", "probability": "Predicted probability",
    },
    "es": {
        "cm_title": "Matriz de confusión", "actual": "Real", "predicted": "Predicho",
        "roc_title": "Curva ROC", "tpr": "Sensibilidad (tasa de verdaderos positivos)", "fpr": "1 − especificidad (tasa de falsos positivos)",
        "chance": "Azar", "model": "Modelo",
        "f1_title": "F1 por clase", "f1": "F1",
        "scatter_title": "Predicho frente a real", "error_title": "Errores de predicción",
        "error": "Error (predicho − real)", "count": "Registros", "perfect": "Predicción perfecta",
        "importance_title": "Importancia de las variables", "importance": "Importancia (pérdida de rendimiento al desordenarla)",
        "pred_dist_title": "Distribución de las predicciones", "pred_value": "Valor predicho",
        "pred_box_title": "Dispersión de las predicciones",
        "class_dist_title": "Clases predichas", "class": "Clase",
        "confidence_title": "Probabilidad predicha por clase", "probability": "Probabilidad predicha",
    },
    "fr": {
        "cm_title": "Matrice de confusion", "actual": "Réel", "predicted": "Prédit",
        "roc_title": "Courbe ROC", "tpr": "Sensibilité (taux de vrais positifs)", "fpr": "1 − spécificité (taux de faux positifs)",
        "chance": "Hasard", "model": "Modèle",
        "f1_title": "F1 par classe", "f1": "F1",
        "scatter_title": "Prédit vs réel", "error_title": "Erreurs de prédiction",
        "error": "Erreur (prédit − réel)", "count": "Enregistrements", "perfect": "Prédiction parfaite",
        "importance_title": "Importance des variables", "importance": "Importance (perte de score si mélangée)",
        "pred_dist_title": "Distribution des prédictions", "pred_value": "Valeur prédite",
        "pred_box_title": "Dispersion des prédictions",
        "class_dist_title": "Classes prédites", "class": "Classe",
        "confidence_title": "Probabilité prédite par classe", "probability": "Probabilité prédite",
    },
}


def use_language(lang):
    """Set the figure language for the current thread/request."""
    _LANG.set(lang if lang in LABELS else "en")


def L(key):
    return LABELS[_LANG.get()].get(key, LABELS["en"][key])


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_base64(fig, save_path=None):
    if save_path:
        fig.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _ink_for(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.55 else INK


def _seq_color(fraction):
    idx = int(round(fraction * (len(SEQ) - 1)))
    return SEQ[max(0, min(len(SEQ) - 1, idx))]


# ── Training figures ──────────────────────────────────────────────────────────
def confusion_matrix_figure(matrix, labels, save_path=None):
    matrix = np.asarray(matrix)
    n = len(labels)
    size = max(4.2, 1.1 * n + 2.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    ax.grid(False)
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    shares = matrix / row_totals
    for i in range(n):
        for j in range(n):
            color = _seq_color(shares[i, j])
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, color=color, linewidth=0))
            ink = _ink_for(color)
            ax.text(j + 0.48, i + 0.42, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=13, fontweight="semibold", color=ink)
            ax.text(j + 0.48, i + 0.66, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=9, color=ink)
    ax.set_xlim(-0.02, n)
    ax.set_ylim(n, -0.02)
    ax.set_xticks([k + 0.48 for k in range(n)], [str(l) for l in labels])
    ax.set_yticks([k + 0.48 for k in range(n)], [str(l) for l in labels])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(L("predicted"))
    ax.set_ylabel(L("actual"))
    ax.set_title(L("cm_title"))
    return to_base64(fig, save_path)


def roc_figure(fpr, tpr, auc, save_path=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label=L("chance"))
    ax.fill_between(fpr, tpr, color=PRIMARY, alpha=0.10, linewidth=0)
    ax.plot(fpr, tpr, color=PRIMARY, label=f"{L('model')} (AUC {auc:.3f})" if auc is not None else L("model"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(L("fpr"))
    ax.set_ylabel(L("tpr"))
    ax.set_title(L("roc_title"))
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    return to_base64(fig, save_path)


def f1_by_class_figure(labels, scores, save_path=None):
    fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(labels) + 2.5), 3.8))
    ax.grid(axis="x", visible=False)
    bars = ax.bar([str(l) for l in labels], scores, color=PRIMARY, width=0.5)
    for bar, value in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(L("f1"))
    ax.set_title(L("f1_title"))
    return to_base64(fig, save_path)


def scatter_figure(y_true, y_pred, rmse=None, save_path=None):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lo, hi = float(np.nanmin([y_true.min(), y_pred.min()])), float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, label=L("perfect"))
    ax.scatter(y_true, y_pred, s=22, color=PRIMARY, alpha=0.55, edgecolors=SURFACE, linewidths=0.8)
    ax.set_xlabel(L("actual"))
    ax.set_ylabel(L("predicted"))
    title = L("scatter_title") + (f"  ·  RMSE {rmse:.3g}" if rmse is not None else "")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return to_base64(fig, save_path)


def error_histogram_figure(y_true, y_pred, save_path=None):
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.grid(axis="x", visible=False)
    ax.hist(errors, bins=30, color=PRIMARY, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel(L("error"))
    ax.set_ylabel(L("count"))
    ax.set_title(L("error_title"))
    return to_base64(fig, save_path)


def importance_figure(names, values, save_path=None, top=15):
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=True)[:top][::-1]
    if not pairs:
        return None
    labels, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.34 * len(pairs) + 1.2)))
    ax.grid(axis="y", visible=False)
    colors = [PRIMARY if v >= 0 else GRID for v in vals]
    ax.barh(labels, vals, color=colors, height=0.6)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel(L("importance"))
    ax.set_title(L("importance_title"))
    return to_base64(fig, save_path)


# ── Prediction figures ────────────────────────────────────────────────────────
def prediction_histogram_figure(values):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.grid(axis="x", visible=False)
    ax.hist(np.asarray(values, dtype=float), bins=30, color=PRIMARY, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlabel(L("pred_value"))
    ax.set_ylabel(L("count"))
    ax.set_title(L("pred_dist_title"))
    return to_base64(fig)


def prediction_box_figure(values):
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.grid(axis="y", visible=False)
    ax.boxplot(np.asarray(values, dtype=float), vert=False, widths=0.45, patch_artist=True,
               boxprops={"facecolor": "#e8edfc", "edgecolor": PRIMARY},
               medianprops={"color": PRIMARY, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED},
               flierprops={"marker": "o", "markersize": 4, "markerfacecolor": PRIMARY, "markeredgecolor": SURFACE})
    ax.set_yticks([])
    ax.set_xlabel(L("pred_value"))
    ax.set_title(L("pred_box_title"))
    return to_base64(fig)


def class_count_figure(predictions):
    labels, counts = np.unique(np.asarray(predictions).astype(str), return_counts=True)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(labels) + 2.5), 3.8))
    ax.grid(axis="x", visible=False)
    bars = ax.bar(labels, counts, color=PRIMARY, width=0.5)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel(L("class"))
    ax.set_ylabel(L("count"))
    ax.set_title(L("class_dist_title"))
    return to_base64(fig)


def class_probability_figure(proba_df):
    classes = list(proba_df.columns)
    fig, ax = plt.subplots(figsize=(6, max(2.6, 0.6 * len(classes) + 1.4)))
    ax.grid(axis="y", visible=False)
    data = [proba_df[c].to_numpy(dtype=float) for c in classes]
    box = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
                     medianprops={"color": INK, "linewidth": 1.5},
                     whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                     flierprops={"marker": "o", "markersize": 3, "markeredgecolor": SURFACE})
    for i, patch in enumerate(box["boxes"]):
        color = SERIES[i % len(SERIES)] if len(classes) <= len(SERIES) else PRIMARY
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(classes) + 1), [str(c) for c in classes])
    ax.set_xlim(0, 1)
    ax.set_xlabel(L("probability"))
    ax.set_title(L("confidence_title"))
    return to_base64(fig)


# ── Data behind the figures (drawn natively in the browser, read by the assistant) ──
def _downsample_curve(xs, ys, thresholds, max_points=120):
    if len(xs) <= max_points:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, max_points).astype(int))
    return ([round(float(xs[i]), 4) for i in idx], [round(float(ys[i]), 4) for i in idx],
            [None if not np.isfinite(thresholds[i]) else round(float(thresholds[i]), 4) for i in idx])


def classification_plot_data(y_true, y_pred, labels, y_proba_pos=None, pos_label=None):
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    data = {"labels": [str(l) for l in labels], "confusion": matrix.tolist()}
    if y_proba_pos is not None and pos_label is not None and len(labels) == 2:
        fpr, tpr, thr = roc_curve(y_true, y_proba_pos, pos_label=pos_label)
        fx, ty, th = _downsample_curve(fpr, tpr, thr)
        y_bin = (np.asarray(y_true) == pos_label).astype(int)
        data["roc"] = {"fpr": fx, "tpr": ty, "threshold": th,
                       "auc": round(float(roc_auc_score(y_bin, y_proba_pos)), 4),
                       "positive_label": str(pos_label)}
    return data


def regression_plot_data(y_true, y_pred, max_points=600, bins=30, seed=0):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    idx = np.arange(len(y_true))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    errors = y_pred - y_true
    counts, edges = np.histogram(errors, bins=bins)
    return {
        "y_true": [round(float(v), 4) for v in y_true[idx]],
        "y_pred": [round(float(v), 4) for v in y_pred[idx]],
        "error_hist": {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]},
        "n": int(len(y_true)),
    }


def plot_data_as_text(plot_data):
    """Plain-text version of plot_data, for a text-only assistant model."""
    if not plot_data:
        return ""
    lines = []
    if "confusion" in plot_data:
        labels = plot_data["labels"]
        lines.append("Confusion matrix on out-of-fold predictions (rows = actual class, columns = predicted class):")
        lines.append("actual \\ predicted | " + " | ".join(labels))
        for label, row in zip(labels, plot_data["confusion"]):
            lines.append(f"{label} | " + " | ".join(str(v) for v in row))
    roc = plot_data.get("roc")
    if roc:
        lines.append(f"ROC curve for positive class '{roc['positive_label']}': AUC = {roc['auc']}.")
        points = [(f, t) for f, t in zip(roc["fpr"], roc["tpr"])]
        for target in (0.05, 0.1, 0.2):
            best = max((t for f, t in points if f <= target), default=None)
            if best is not None:
                lines.append(f"Sensitivity at false positive rate <= {target:.0%}: {best:.2f}")
    if "error_hist" in plot_data:
        y_true, y_pred = np.asarray(plot_data["y_true"]), np.asarray(plot_data["y_pred"])
        err = y_pred - y_true
        lines.append(f"Regression errors (sample of {len(err)} of {plot_data.get('n')} records): "
                     f"mean error {err.mean():.4g}, median absolute error {np.median(np.abs(err)):.4g}, "
                     f"5th-95th percentile of error {np.percentile(err, 5):.4g} to {np.percentile(err, 95):.4g}.")
    return "\n".join(lines)
