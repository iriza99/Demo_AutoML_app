"""Generic detectors for modelling pitfalls in tabular clinical datasets.

These detectors only look at the data, the selected target, the training
configuration and (optionally) a data dictionary. They never read the study
answer key, so the same code can be evaluated on any dataset.

Issue types and the competence dimension needed to spot them:

    imbalance_metric  (ml)      rare outcome evaluated with accuracy
    identifier        (ml)      row identifier carrying outcome information
    temporal_leak     (domain)  variable not available at prediction time
    unit_mixing       (domain)  one variable recorded in two unit systems
"""
import math
import re

import numpy as np
import pandas as pd

from utils.typed_decision import safe_call

ISSUE_DIMENSION = {
    "imbalance_metric": "ml",
    "identifier": "ml",
    "temporal_leak": "domain",
    "unit_mixing": "domain",
}

# Minority share below which accuracy becomes misleading
IMBALANCE_THRESHOLD = 0.20
# Single-feature AUC above which a column is "too good to be true"
LEAK_AUC_THRESHOLD = 0.85
# Share of total importance for a feature to count as dominant after training
DOMINANT_SHARE = 0.30

# Common clinical conversion factors: (units, analyte, factor).
# Only the absolute log-ratio between the two modes is compared.
UNIT_FACTORS = [
    ("mg/dL ↔ µmol/L", "creatinine", 88.4),
    ("mmol/L ↔ mg/dL", "lactate", 9.01),
    ("mmol/L ↔ mg/dL", "glucose", 18.0),
    ("mmol/L ↔ mg/dL", "cholesterol", 38.67),
    ("mg/dL ↔ µmol/L", "bilirubin", 17.1),
    ("mmol/L ↔ mg/dL", "urea", 6.0),
    ("g/dL ↔ g/L", "hemoglobin", 10.0),
    ("×10", "any", 10.0),
    ("×1000", "any", 1000.0),
]
UNIT_LOG_TOLERANCE = 0.2

ID_NAME_PATTERN = re.compile(
    r"(^|_)(id|ids|record|episode|patient|nhc|mrn|code|codigo|numero|number|row|index)($|_)",
    re.IGNORECASE,
)
# Fallback cues used when no decision layer is configured. They describe
# information that is usually recorded after the prediction moment.
LATE_INFO_PATTERN = re.compile(
    r"follow[\s_-]?up|seguimiento|revisi[oó]n|post[\s_-]|after|despu[eé]s|posterior|"
    r"administered|administrad|given|started|iniciad|treatment|tratamiento|"
    r"outcome|resultado final",
    re.IGNORECASE,
)


# ── Dataset profile (target independent) ─────────────────────────────────────

def is_identifier(series, name):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().mean() > 0.01:
        return bool(ID_NAME_PATTERN.search(name)) and series.nunique() > 0.95 * len(series)
    values = numeric.dropna()
    if len(values) < 20 or not np.allclose(values, values.round()):
        return False
    unique_ratio = values.nunique() / len(values)
    if unique_ratio < 0.95:
        return False
    span_ratio = (values.max() - values.min() + 1) / len(values)
    return bool(ID_NAME_PATTERN.search(name)) or span_ratio < 1.5


def _two_gaussians(x, iters=200):
    """Fit a 1-D two-component Gaussian mixture with EM."""
    mu = np.percentile(x, [20, 80]).astype(float)
    sd = np.array([x.std(), x.std()]) / 2 + 1e-6
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        dens = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
        total = dens.sum(axis=1, keepdims=True) + 1e-300
        resp = dens / total
        nk = resp.sum(axis=0) + 1e-9
        w = nk / len(x)
        mu = (resp * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-6
    loglik = np.log(total).sum()
    return w, mu, sd, loglik


def detect_unit_mixing(series):
    """Return details if a numeric column looks like two unit systems mixed."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if len(values) < 50 or values.nunique() < 20:
        return None
    x = np.log(values.to_numpy(dtype=float))
    w, mu, sd, loglik2 = _two_gaussians(x)
    loglik1 = (-0.5 * np.log(2 * np.pi * x.var()) - 0.5) * len(x)
    # BIC: the mixture has 3 extra parameters
    if (loglik2 - loglik1) < 1.5 * 3 * math.log(len(x)):
        return None
    if w.min() < 0.08:
        return None
    delta = abs(mu[1] - mu[0])
    if delta < 2.5 * sd.max():
        return None
    for label, analyte, factor in UNIT_FACTORS:
        if abs(delta - math.log(factor)) < UNIT_LOG_TOLERANCE:
            low, high = sorted(np.exp(mu))
            minor_share = float(w.min())
            return {
                "factor_label": label,
                "typical_analyte": analyte,
                "factor": factor,
                "observed_ratio": round(float(math.exp(delta)), 2),
                "low": round(float(low), 2),
                "high": round(float(high), 2),
                "minor_share": round(minor_share, 3),
            }
    return None


def profile_dataset(df):
    profile = {"n_rows": int(len(df)), "columns": list(map(str, df.columns)),
               "identifiers": [], "unit_mixing": {}}
    for col in df.columns:
        series = df[col]
        if is_identifier(series, str(col)):
            profile["identifiers"].append(str(col))
            continue
        if pd.api.types.is_numeric_dtype(series):
            found = detect_unit_mixing(series)
            if found:
                profile["unit_mixing"][str(col)] = found
    return profile


# ── Target dependent checks ──────────────────────────────────────────────────

def _rank_auc(score, y):
    mask = score.notna() & y.notna()
    score, y = score[mask].astype(float), y[mask].astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = score.rank()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


def single_feature_auc(df, target):
    """AUC of each column on its own for a binary target (rare class = 1)."""
    counts = df[target].value_counts()
    if len(counts) != 2:
        return {}
    y = (df[target] == counts.idxmin()).astype(int)
    result = {}
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            score = series
        elif series.nunique() <= 50:
            # Target-mean encoding; fine for low-cardinality categoricals
            score = series.map(y.groupby(series).mean())
        else:
            continue
        result[str(col)] = round(_rank_auc(score, y), 4)
    return result


def target_info(df, target):
    counts = df[target].value_counts(dropna=True)
    info = {"target": target, "n_classes": int(len(counts))}
    if 2 <= len(counts) <= 20:
        shares = counts / counts.sum()
        info.update({
            "minority_class": str(shares.idxmin()),
            "minority_share": round(float(shares.min()), 4),
            "majority_share": round(float(shares.max()), 4),
        })
    if len(counts) == 2:
        info["single_feature_auc"] = single_feature_auc(df, target)
    return info


def semantic_leak_candidates(dictionary, lang="en", decision_layer=None):
    """Columns whose description suggests they are recorded after prediction.

    Returns {column: probability}. Uses the decision layer when available,
    otherwise a conservative keyword fallback.
    """
    if not dictionary:
        return {}
    moment = dictionary.get("prediction_moment", {})
    moment = moment.get(lang) or moment.get("en") or ""
    target = dictionary.get("target")
    result = {}
    for col, desc in dictionary.get("columns", {}).items():
        if col == target:
            continue
        text = desc.get(lang) or desc.get("en") or ""
        prob = None
        if decision_layer is not None:
            question = (f"A prediction is made {moment.lower()}. Is the value of the "
                        f"variable '{col}' ({text}) already known at that moment?")
            p_known = safe_call(decision_layer.noul, {"variable": col, "description": text}, question)
            if p_known is not None:
                prob = 1.0 - p_known
        if prob is None:
            prob = 0.7 if LATE_INFO_PATTERN.search(f"{col} {text}") else 0.0
        if prob >= 0.5:
            result[col] = round(prob, 3)
    return result


# ── Issue evaluation ─────────────────────────────────────────────────────────

def _issue(kind, column=None, severity="medium", **facts):
    key = f"{kind}:{column}" if column else kind
    return {"key": key, "type": kind, "dimension": ISSUE_DIMENSION[kind],
            "column": column, "severity": severity, "facts": facts}


def evaluate(stage, state, profile, tinfo, semantic):
    """List the open issues for the current UI state.

    stage: "data" (dataset loaded), "pre_train" or "post_train".
    state: {"dropped": [...], "target": str, "metric": str,
            "feature_importance": {col: value}, "metrics": {name: value}}
    """
    dropped = set(state.get("dropped") or [])
    target = state.get("target")
    issues = []

    for col, facts in profile.get("unit_mixing", {}).items():
        if col not in dropped and col != target:
            issues.append(_issue("unit_mixing", col, "high", **facts))

    if stage == "data" or not tinfo:
        return issues

    identifiers = [c for c in profile.get("identifiers", []) if c not in dropped and c != target]
    aucs = tinfo.get("single_feature_auc", {})
    leaks = {}
    for col, auc in aucs.items():
        if col in dropped or col in profile.get("identifiers", []):
            continue
        if auc >= LEAK_AUC_THRESHOLD or col in semantic:
            leaks[col] = {"single_feature_auc": auc, "semantic_prob": semantic.get(col, 0.0)}

    if stage == "pre_train":
        for col in identifiers:
            issues.append(_issue("identifier", col, "high",
                                 single_feature_auc=aucs.get(col)))
        for col, facts in leaks.items():
            both = facts["single_feature_auc"] >= LEAK_AUC_THRESHOLD and facts["semantic_prob"] > 0
            issues.append(_issue("temporal_leak", col, "high" if both else "medium", **facts))

    minority = tinfo.get("minority_share")
    metric = (state.get("metric") or "accuracy").lower()
    if minority is not None and minority < IMBALANCE_THRESHOLD and metric == "accuracy":
        facts = {"minority_share": minority, "baseline_accuracy": tinfo.get("majority_share")}
        acc = (state.get("metrics") or {}).get("accuracy")
        if stage == "post_train" and acc is not None:
            facts["accuracy"] = acc
        issues.append(_issue("imbalance_metric", None, "high", **facts))

    if stage == "post_train":
        importance = {k: v for k, v in (state.get("feature_importance") or {}).items()
                      if v is not None and v > 0}
        total = sum(importance.values())
        if total > 0:
            ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
            for rank, (col, value) in enumerate(ranked[:3], start=1):
                share = value / total
                if col in identifiers and (share >= DOMINANT_SHARE or rank == 1):
                    issues.append(_issue("identifier", col, "high", importance_rank=rank,
                                         importance_share=round(share, 3)))
                elif col in leaks and (share >= DOMINANT_SHARE or rank == 1):
                    issues.append(_issue("temporal_leak", col, "high", importance_rank=rank,
                                         importance_share=round(share, 3), **leaks[col]))
    return issues
