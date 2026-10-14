"""Generate the two synthetic clinical datasets used in the user study.

Each dataset carries four planted traps (two ML pitfalls, two clinical
pitfalls) that are matched across datasets so the two task conditions are
equivalent. See study/TRAPS.md for the rationale and the scoring rubric.

The trap manifests written to study/answer_key/ are for the researchers. The application never reads them: guardrail detectors in
utils/guardrails.py must find the traps from the data and data dictionary
alone (this is checked in tests/test_guardrails.py).

Usage:
    python study/generate_datasets.py            # writes study/datasets/
    python study/generate_datasets.py --check    # also prints trap strength
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(STUDY_DIR, "datasets")
ANSWER_KEY_DIR = os.path.join(STUDY_DIR, "answer_key")
N_ROWS = 1600

# Conversion factors used for the unit-mixing trap
CREATININE_MGDL_TO_UMOLL = 88.4
LACTATE_MMOLL_TO_MGDL = 9.01


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(linear, prevalence):
    """Find the intercept that gives the requested outcome prevalence."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(linear + mid).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _leaky_ids(rng, y, start, strength=1.0, noise=2.0):
    """Row identifiers that were assigned after sorting by outcome.

    Mimics an export where positive cases were appended from a separate
    registry, so the identifier carries outcome information.
    """
    key = y * strength + rng.uniform(0, noise, size=len(y))
    ranks = np.argsort(np.argsort(key))
    return start + ranks


def _yes_no(arr):
    return np.where(np.asarray(arr).astype(bool), "yes", "no")


def make_heart_failure(seed=11):
    """Dataset A: 30-day readmission after a heart-failure admission.

    Prediction moment: hospital discharge.
    """
    rng = np.random.default_rng(seed)
    n = N_ROWS
    age = np.clip(rng.normal(76, 9, n), 45, 98).round()
    female = rng.binomial(1, 0.48, n)
    lvef = np.clip(rng.normal(40, 13, n), 10, 75).round()
    nyha = rng.choice([2, 3, 4], size=n, p=[0.45, 0.40, 0.15])
    sbp = np.clip(rng.normal(125, 20, n), 75, 210).round()
    hr = np.clip(rng.normal(82, 15, n), 40, 160).round()
    sodium = np.clip(rng.normal(138, 4, n), 120, 150).round()
    creat_mgdl = np.exp(rng.normal(np.log(1.2), 0.35, n))
    hb = np.clip(rng.normal(12.3, 1.8, n), 6.5, 18).round(1)
    ntprobnp_adm = np.exp(rng.normal(np.log(4000), 0.9, n))
    diabetes = rng.binomial(1, 0.38, n)
    copd = rng.binomial(1, 0.22, n)
    af = rng.binomial(1, 0.42, n)
    prior_adm = rng.poisson(0.6, n)
    los = rng.poisson(6, n) + 1
    bb = rng.binomial(1, 0.75, n)
    acei = rng.binomial(1, 0.65, n)
    # Hidden: hospital site. Site 1 reports creatinine in umol/L and has a
    # higher baseline readmission rate (different case mix / follow-up).
    site = rng.binomial(1, 0.4, n)

    linear = 1.7 * (
        0.03 * (age - 76) + 0.35 * (nyha - 2) - 0.015 * (lvef - 40)
        + 0.06 * (138 - sodium) + 0.6 * np.log(creat_mgdl / 1.2)
        + 0.35 * np.log(ntprobnp_adm / 4000) + 0.35 * prior_adm
        + 0.25 * diabetes + 0.25 * copd - 0.2 * bb - 0.2 * acei
        + 0.04 * (los - 6) - 0.1 * (hb - 12.3) + 0.5 * site
    )
    intercept = _calibrate_intercept(linear, 0.12)
    y = rng.binomial(1, _sigmoid(linear + intercept))

    # TRAP C1 (temporal leakage): NT-proBNP measured at the outpatient
    # follow-up visit, i.e. after discharge and often after the readmission.
    base = np.where(y == 1, np.log(3000), np.log(900)) + 0.3 * np.log(ntprobnp_adm / 4000)
    ntprobnp_fu = np.exp(rng.normal(base, 0.6))
    ntprobnp_fu = ntprobnp_fu.round()
    ntprobnp_fu[rng.random(n) < 0.08] = np.nan

    # TRAP C2 (unit mixing): site 1 creatinine in umol/L
    creatinine = np.where(site == 1, (creat_mgdl * CREATININE_MGDL_TO_UMOLL).round(),
                          creat_mgdl.round(2))

    df = pd.DataFrame({
        # TRAP M2 (identifier leakage)
        "record_id": _leaky_ids(rng, y, start=20001),
        "age": age.astype(int),
        "sex": np.where(female == 1, "F", "M"),
        "lvef_percent": lvef.astype(int),
        "nyha_class": nyha,
        "systolic_bp": sbp.astype(int),
        "heart_rate": hr.astype(int),
        "sodium": sodium.astype(int),
        "creatinine": creatinine,
        "hemoglobin": hb,
        "nt_probnp_admission": ntprobnp_adm.round().astype(int),
        "diabetes": _yes_no(diabetes),
        "copd": _yes_no(copd),
        "atrial_fibrillation": _yes_no(af),
        "prior_hf_admissions_12m": prior_adm,
        "length_of_stay_days": los,
        "discharge_beta_blocker": _yes_no(bb),
        "discharge_acei_arni": _yes_no(acei),
        "nt_probnp_followup": ntprobnp_fu,
        # TRAP M1 (class imbalance + accuracy as default metric): ~12 % positives
        "readmission_30d": _yes_no(y),
    })
    hidden = {"site": site, "y": y}
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), hidden


def make_sepsis(seed=23):
    """Dataset B: sepsis within 48 h in ED patients with suspected infection.

    Prediction moment: emergency department triage.
    """
    rng = np.random.default_rng(seed)
    n = N_ROWS
    age = np.clip(rng.normal(64, 18, n), 18, 99).round()
    female = rng.binomial(1, 0.5, n)
    temp = np.clip(rng.normal(37.8, 0.9, n), 34.5, 41.5).round(1)
    hr = np.clip(rng.normal(98, 18, n), 45, 180).round()
    rr = np.clip(rng.normal(20, 5, n), 10, 45).round()
    sbp = np.clip(rng.normal(122, 24, n), 60, 210).round()
    gcs = np.where(rng.random(n) < 0.85, 15, rng.integers(9, 15, n))
    spo2 = np.clip(rng.normal(95, 3, n), 75, 100).round()
    wbc = np.exp(rng.normal(np.log(11), 0.45, n)).round(1)
    platelets = np.clip(rng.normal(230, 80, n), 20, 600).round()
    lactate_mmol = np.exp(rng.normal(np.log(1.8), 0.45, n))
    crp = np.exp(rng.normal(np.log(80), 0.9, n)).round()
    creat = np.exp(rng.normal(np.log(1.1), 0.4, n)).round(2)
    diabetes = rng.binomial(1, 0.28, n)
    immuno = rng.binomial(1, 0.12, n)
    nursing_home = rng.binomial(1, 0.10, n)
    source = rng.choice(["respiratory", "urinary", "abdominal", "skin", "unknown"],
                        size=n, p=[0.38, 0.27, 0.15, 0.10, 0.10])
    # Hidden: hospital site. Site 1 reports lactate in mg/dL.
    site = rng.binomial(1, 0.4, n)

    linear = 1.4 * (
        0.02 * (age - 64) + 0.02 * (hr - 98) + 0.08 * (rr - 20) - 0.02 * (sbp - 122)
        + 0.25 * (15 - gcs) - 0.08 * (spo2 - 95) + 0.9 * np.log(lactate_mmol / 1.8)
        + 0.3 * np.log(crp / 80) + 0.4 * np.abs(np.log(wbc / 11))
        + 0.4 * np.log(creat / 1.1) + 0.5 * immuno + 0.4 * nursing_home + 0.4 * site
        + 0.3 * (source == "abdominal")
    )
    intercept = _calibrate_intercept(linear, 0.10)
    y = rng.binomial(1, _sigmoid(linear + intercept))

    # TRAP C1 (treatment as label proxy): IV antibiotics are given in the ED
    # *because* the clinician already suspects sepsis, after triage.
    z = (linear - linear.mean()) / linear.std()
    p_abx = np.where(y == 1, 0.88, _sigmoid(-2.2 + 0.6 * z))
    abx = rng.binomial(1, p_abx)

    # TRAP C2 (unit mixing): site 1 lactate in mg/dL
    lactate = np.where(site == 1, (lactate_mmol * LACTATE_MMOLL_TO_MGDL).round(1),
                       lactate_mmol.round(1))

    df = pd.DataFrame({
        # TRAP M2 (identifier leakage)
        "episode_id": _leaky_ids(rng, y, start=50001),
        "age": age.astype(int),
        "sex": np.where(female == 1, "F", "M"),
        "temperature": temp,
        "heart_rate": hr.astype(int),
        "respiratory_rate": rr.astype(int),
        "systolic_bp": sbp.astype(int),
        "glasgow_coma_scale": gcs,
        "spo2": spo2.astype(int),
        "wbc": wbc,
        "platelets": platelets.astype(int),
        "lactate": lactate,
        "crp": crp.astype(int),
        "creatinine": creat,
        "diabetes": _yes_no(diabetes),
        "immunosuppression": _yes_no(immuno),
        "nursing_home_resident": _yes_no(nursing_home),
        "suspected_source": source,
        "antibiotic_iv_ed": _yes_no(abx),
        # TRAP M1 (class imbalance + accuracy as default metric): ~10 % positives
        "sepsis_48h": _yes_no(y),
    })
    hidden = {"site": site, "y": y}
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), hidden


DATASETS = {
    "A": {
        "name": "hf_readmission",
        "builder": make_heart_failure,
        "target": "readmission_30d",
        "positive": "yes",
        "prediction_moment": {
            "es": "En el momento del alta hospitalaria",
            "en": "At hospital discharge",
        },
        "task": {
            "es": ("Entrena un modelo que, en el momento del alta, prediga qué pacientes "
                   "ingresados por insuficiencia cardiaca reingresarán en 30 días. "
                   "Decide al final si lo desplegarías en tu servicio."),
            "en": ("Train a model that, at discharge, predicts which heart-failure "
                   "patients will be readmitted within 30 days. At the end, decide "
                   "whether you would deploy it in your department."),
        },
        "dictionary": {
            "record_id": ("Identificador del registro", "Record identifier"),
            "age": ("Edad (años)", "Age (years)"),
            "sex": ("Sexo", "Sex"),
            "lvef_percent": ("Fracción de eyección del VI (%)", "Left ventricular ejection fraction (%)"),
            "nyha_class": ("Clase funcional NYHA al alta", "NYHA functional class at discharge"),
            "systolic_bp": ("Presión arterial sistólica al alta (mmHg)", "Systolic blood pressure at discharge (mmHg)"),
            "heart_rate": ("Frecuencia cardiaca al alta (lpm)", "Heart rate at discharge (bpm)"),
            "sodium": ("Sodio sérico (mmol/L)", "Serum sodium (mmol/L)"),
            "creatinine": ("Creatinina sérica al alta", "Serum creatinine at discharge"),
            "hemoglobin": ("Hemoglobina (g/dL)", "Hemoglobin (g/dL)"),
            "nt_probnp_admission": ("NT-proBNP al ingreso (pg/mL)", "NT-proBNP at admission (pg/mL)"),
            "diabetes": ("Diabetes mellitus", "Diabetes mellitus"),
            "copd": ("EPOC", "COPD"),
            "atrial_fibrillation": ("Fibrilación auricular", "Atrial fibrillation"),
            "prior_hf_admissions_12m": ("Ingresos por IC en los 12 meses previos", "HF admissions in the previous 12 months"),
            "length_of_stay_days": ("Estancia hospitalaria (días)", "Length of stay (days)"),
            "discharge_beta_blocker": ("Betabloqueante prescrito al alta", "Beta-blocker prescribed at discharge"),
            "discharge_acei_arni": ("IECA/ARA-II/ARNI prescrito al alta", "ACEi/ARB/ARNI prescribed at discharge"),
            "nt_probnp_followup": ("NT-proBNP en la revisión ambulatoria (pg/mL)", "NT-proBNP at the outpatient follow-up visit (pg/mL)"),
            "readmission_30d": ("Reingreso por cualquier causa en 30 días", "All-cause readmission within 30 days"),
        },
    },
    "B": {
        "name": "ed_sepsis",
        "builder": make_sepsis,
        "target": "sepsis_48h",
        "positive": "yes",
        "prediction_moment": {
            "es": "En el triaje de urgencias",
            "en": "At emergency department triage",
        },
        "task": {
            "es": ("Entrena un modelo que, en el triaje de urgencias, prediga qué pacientes "
                   "con sospecha de infección desarrollarán sepsis en 48 horas. "
                   "Decide al final si lo desplegarías en tu servicio."),
            "en": ("Train a model that, at ED triage, predicts which patients with "
                   "suspected infection will develop sepsis within 48 hours. At the end, "
                   "decide whether you would deploy it in your department."),
        },
        "dictionary": {
            "episode_id": ("Identificador del episodio", "Episode identifier"),
            "age": ("Edad (años)", "Age (years)"),
            "sex": ("Sexo", "Sex"),
            "temperature": ("Temperatura en triaje (°C)", "Temperature at triage (°C)"),
            "heart_rate": ("Frecuencia cardiaca en triaje (lpm)", "Heart rate at triage (bpm)"),
            "respiratory_rate": ("Frecuencia respiratoria en triaje (rpm)", "Respiratory rate at triage (bpm)"),
            "systolic_bp": ("Presión arterial sistólica en triaje (mmHg)", "Systolic blood pressure at triage (mmHg)"),
            "glasgow_coma_scale": ("Escala de coma de Glasgow", "Glasgow coma scale"),
            "spo2": ("Saturación de oxígeno (%)", "Oxygen saturation (%)"),
            "wbc": ("Leucocitos (x10^9/L)", "White blood cells (x10^9/L)"),
            "platelets": ("Plaquetas (x10^9/L)", "Platelets (x10^9/L)"),
            "lactate": ("Lactato venoso", "Venous lactate"),
            "crp": ("Proteína C reactiva (mg/L)", "C-reactive protein (mg/L)"),
            "creatinine": ("Creatinina sérica (mg/dL)", "Serum creatinine (mg/dL)"),
            "diabetes": ("Diabetes mellitus", "Diabetes mellitus"),
            "immunosuppression": ("Inmunosupresión", "Immunosuppression"),
            "nursing_home_resident": ("Residente en centro sociosanitario", "Nursing home resident"),
            "suspected_source": ("Foco de infección sospechado", "Suspected source of infection"),
            "antibiotic_iv_ed": ("Antibiótico IV administrado en urgencias", "IV antibiotic administered in the ED"),
            "sepsis_48h": ("Sepsis (Sepsis-3) en las primeras 48 h", "Sepsis (Sepsis-3) within 48 h"),
        },
    },
}

# Researcher-only answer key. Matched pairs across datasets share the trap id.
TRAP_MANIFEST = {
    "A": [
        {"id": "M1", "dimension": "ml", "type": "class_imbalance_accuracy", "column": "readmission_30d"},
        {"id": "M2", "dimension": "ml", "type": "identifier_leakage", "column": "record_id"},
        {"id": "C1", "dimension": "domain", "type": "temporal_leakage", "column": "nt_probnp_followup"},
        {"id": "C2", "dimension": "domain", "type": "unit_mixing", "column": "creatinine",
         "units": ["mg/dL", "umol/L"], "factor": CREATININE_MGDL_TO_UMOLL},
    ],
    "B": [
        {"id": "M1", "dimension": "ml", "type": "class_imbalance_accuracy", "column": "sepsis_48h"},
        {"id": "M2", "dimension": "ml", "type": "identifier_leakage", "column": "episode_id"},
        {"id": "C1", "dimension": "domain", "type": "treatment_as_proxy", "column": "antibiotic_iv_ed"},
        {"id": "C2", "dimension": "domain", "type": "unit_mixing", "column": "lactate",
         "units": ["mmol/L", "mg/dL"], "factor": LACTATE_MMOLL_TO_MGDL},
    ],
}


def dictionary_payload(key):
    spec = DATASETS[key]
    return {
        "dataset": spec["name"],
        "target": spec["target"],
        "prediction_moment": spec["prediction_moment"],
        "task": spec["task"],
        "columns": {col: {"es": es, "en": en} for col, (es, en) in spec["dictionary"].items()},
    }


def _auc(score, y):
    """Rank-based ROC AUC without scikit-learn."""
    score = pd.Series(score).astype(float)
    mask = score.notna().to_numpy()
    s, t = score[mask], np.asarray(y)[mask]
    ranks = s.rank().to_numpy()
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    return (ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def check(key, df, hidden):
    """Print trap strength so the design can be audited."""
    spec = DATASETS[key]
    y = hidden["y"]
    print(f"\n=== Dataset {key} ({spec['name']}) — n={len(df)}, prevalence={y.mean():.3f}")
    print(f"  majority-class accuracy: {1 - y.mean():.3f}")
    for trap in TRAP_MANIFEST[key]:
        col = trap["column"]
        if col == spec["target"]:
            continue
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values):
            values = (values == "yes").astype(float)
        auc = _auc(values, (df[spec["target"]] == "yes").astype(int).to_numpy())
        print(f"  {trap['id']} {col}: single-feature AUC={max(auc, 1 - auc):.3f}")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        target = (df[spec["target"]] == "yes").astype(int)
        X = pd.get_dummies(df.drop(columns=[spec["target"]]), drop_first=True).fillna(0)
        leaks = [t["column"] for t in TRAP_MANIFEST[key] if t["id"] in ("M2", "C1")]
        clean = X.drop(columns=[c for c in X.columns if any(c.startswith(l) for l in leaks)])
        for label, data in (("with leaks", X), ("without leaks", clean)):
            lr = LogisticRegression(max_iter=3000)
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            auc = cross_val_score(make_pipeline(StandardScaler(), lr), data, target,
                                  cv=5, scoring="roc_auc").mean()
            print(f"  logistic regression CV AUC {label}: {auc:.3f}")
    except ImportError:
        print("  (install scikit-learn to see model-level AUCs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="print trap strength")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for key, spec in DATASETS.items():
        df, hidden = spec["builder"]()
        base = os.path.join(args.out, spec["name"])
        df.to_csv(base + ".csv", index=False)
        with open(base + "_dictionary.json", "w", encoding="utf-8") as fh:
            json.dump(dictionary_payload(key), fh, ensure_ascii=False, indent=2)
        os.makedirs(ANSWER_KEY_DIR, exist_ok=True)
        key_path = os.path.join(ANSWER_KEY_DIR, spec["name"] + "_traps.json")
        with open(key_path, "w", encoding="utf-8") as fh:
            json.dump({"dataset": spec["name"], "traps": TRAP_MANIFEST[key]}, fh, indent=2)
        print(f"wrote {base}.csv")
        if args.check:
            check(key, df, hidden)


if __name__ == "__main__":
    main()
