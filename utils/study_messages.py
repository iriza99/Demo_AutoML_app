"""Texts for guardrail interventions, competence probes and chat adaptation.

Each issue type has three explanation levels:
    novice        plain language, explains why it matters and what to do
    intermediate  concise, with the key numbers
    expert        one line, for a quick glance
Spanish and English are provided; other UI languages fall back to English.
"""

TITLES = {
    "imbalance_metric": {"es": "Posible desbalance de clases", "en": "Possible class imbalance"},
    "identifier": {"es": "Posible identificador en los datos", "en": "Possible identifier in the data"},
    "temporal_leak": {"es": "¿Información disponible al predecir?", "en": "Information available at prediction time?"},
    "unit_mixing": {"es": "Posible mezcla de unidades", "en": "Possible mixed units"},
}

TEXTS = {
    "imbalance_metric": {
        "novice": {
            "es": ("Solo el {minority_pct} % de los pacientes tiene el evento. Un modelo que dijera "
                   "siempre «no» acertaría el {baseline_pct} % de las veces, así que la accuracy "
                   "(porcentaje de aciertos) puede parecer muy alta aunque el modelo no sirva. "
                   "Abre «Opciones avanzadas» y elige ROC AUC o F1 como métrica."),
            "en": ("Only {minority_pct}% of patients have the event. A model that always said "
                   "\"no\" would be right {baseline_pct}% of the time, so accuracy (percentage of "
                   "correct answers) can look very high even if the model is useless. Open "
                   "\"Advanced options\" and choose ROC AUC or F1 as the metric."),
        },
        "intermediate": {
            "es": ("Clase minoritaria: {minority_pct} %. Con accuracy, el modelo trivial ya obtiene "
                   "un {baseline_pct} %. Considera ROC AUC o F1 macro."),
            "en": ("Minority class: {minority_pct}%. With accuracy, the trivial model already "
                   "scores {baseline_pct}%. Consider ROC AUC or macro F1."),
        },
        "expert": {
            "es": "Desbalance {minority_pct} % · accuracy del modelo trivial {baseline_pct} %.",
            "en": "Imbalance {minority_pct}% · trivial-model accuracy {baseline_pct}%.",
        },
    },
    "identifier": {
        "novice": {
            "es": ("La columna «{column}» parece un número de registro, no una característica del "
                   "paciente. Aun así, el modelo puede usarla para «adivinar» el resultado si los "
                   "números se asignaron en un orden relacionado con el desenlace, y eso no "
                   "funcionaría con pacientes nuevos. Te recomendamos excluirla en el paso 2."),
            "en": ("The column \"{column}\" looks like a record number, not a patient "
                   "characteristic. The model can still use it to \"guess\" the outcome if the "
                   "numbers were assigned in an order related to it, which would not work for new "
                   "patients. We recommend excluding it in step 2."),
        },
        "intermediate": {
            "es": ("«{column}» parece un identificador (valores únicos y consecutivos){auc_clause}. "
                   "Probable fuga de información: exclúyela."),
            "en": ("\"{column}\" looks like an identifier (unique, consecutive values){auc_clause}. "
                   "Likely leakage: exclude it."),
        },
        "expert": {
            "es": "«{column}»: identificador{auc_clause}.",
            "en": "\"{column}\": identifier{auc_clause}.",
        },
    },
    "temporal_leak": {
        "novice": {
            "es": ("Comprueba si «{column}» ({description}) se conoce realmente {moment}. Si se "
                   "registra después, el modelo estaría usando información del futuro: aquí "
                   "parecerá muy bueno, pero no se podrá usar en la práctica. Si es así, exclúyela "
                   "en el paso 2."),
            "en": ("Check whether \"{column}\" ({description}) is really known {moment}. If it "
                   "is recorded later, the model would be using information from the future: it "
                   "will look very good here but cannot be used in practice. If so, exclude it in "
                   "step 2."),
        },
        "intermediate": {
            "es": ("«{column}» predice sospechosamente bien{auc_clause}. ¿Está disponible "
                   "{moment}? Si no, exclúyela."),
            "en": ("\"{column}\" predicts suspiciously well{auc_clause}. Is it available "
                   "{moment}? If not, exclude it."),
        },
        "expert": {
            "es": "«{column}»: posible fuga temporal{auc_clause}.",
            "en": "\"{column}\": possible temporal leakage{auc_clause}.",
        },
    },
    "unit_mixing": {
        "novice": {
            "es": ("Los valores de «{column}» forman dos grupos separados (en torno a {low} y a "
                   "{high}, unas {observed_ratio} veces mayor). Suele indicar que parte de los "
                   "datos se registró en otra unidad ({factor_label}). Si es así, el modelo mezcla "
                   "valores no comparables. Revísalo con quien generó los datos o excluye la variable."),
            "en": ("The values of \"{column}\" form two separate groups (around {low} and "
                   "{high}, about {observed_ratio} times larger). This usually means part of the "
                   "data was recorded in another unit ({factor_label}). If so, the model mixes "
                   "values that are not comparable. Check with whoever produced the data or "
                   "exclude the variable."),
        },
        "intermediate": {
            "es": ("«{column}» es bimodal con una razón ≈{observed_ratio} entre modos ({low} frente "
                   "a {high}), compatible con {factor_label}."),
            "en": ("\"{column}\" is bimodal with a ≈{observed_ratio} ratio between modes ({low} "
                   "vs {high}), consistent with {factor_label}."),
        },
        "expert": {
            "es": "«{column}»: posible mezcla de unidades ({factor_label}).",
            "en": "\"{column}\": possible mixed units ({factor_label}).",
        },
    },
}

POST_TRAIN_SUFFIX = {
    "identifier": {
        "es": " Tras el entrenamiento, es la variable n.º {importance_rank} en importancia.",
        "en": " After training, it is the #{importance_rank} most important feature.",
    },
    "temporal_leak": {
        "es": " Tras el entrenamiento, es la variable n.º {importance_rank} en importancia.",
        "en": " After training, it is the #{importance_rank} most important feature.",
    },
    "imbalance_metric": {
        "es": " Accuracy obtenida: {accuracy_pct} %.",
        "en": " Accuracy obtained: {accuracy_pct}%.",
    },
}

PROBES = {
    "imbalance_metric": {
        "question": {
            "es": ("Solo el {minority_pct} % de los pacientes tiene el evento. Si un modelo dijera "
                   "«no» a todos, ¿qué accuracy tendría?"),
            "en": ("Only {minority_pct}% of patients have the event. If a model said \"no\" to "
                   "everyone, what accuracy would it get?"),
        },
        "options": [
            ("correct", {"es": "≈{baseline_pct} %", "en": "≈{baseline_pct}%"}),
            ("wrong", {"es": "≈50 %", "en": "≈50%"}),
            ("dont_know", {"es": "No lo sé", "en": "I don't know"}),
        ],
    },
    "identifier": {
        "question": {
            "es": "¿Qué representa la columna «{column}» y debería usarse para predecir?",
            "en": "What does the column \"{column}\" represent, and should it be used to predict?",
        },
        "options": [
            ("correct", {"es": "Es un identificador; no debería usarse",
                         "en": "It is an identifier; it should not be used"}),
            ("wrong", {"es": "Es una variable útil; debería usarse",
                       "en": "It is a useful variable; it should be used"}),
            ("dont_know", {"es": "No lo sé", "en": "I don't know"}),
        ],
    },
    "temporal_leak": {
        "question": {
            "es": "¿El valor de «{column}» ({description}) se conoce ya {moment}?",
            "en": "Is the value of \"{column}\" ({description}) already known {moment}?",
        },
        "options": [
            ("wrong", {"es": "Sí, ya se conoce", "en": "Yes, it is already known"}),
            ("correct", {"es": "No, se obtiene después", "en": "No, it is obtained later"}),
            ("dont_know", {"es": "No lo sé", "en": "I don't know"}),
        ],
    },
    "unit_mixing": {
        "question": {
            "es": ("«{column}» tiene dos grupos de valores (≈{low} y ≈{high}). "
                   "¿A qué se debe con más probabilidad?"),
            "en": ("\"{column}\" has two groups of values (≈{low} and ≈{high}). "
                   "What is the most likely cause?"),
        },
        "options": [
            ("wrong", {"es": "Hay pacientes leves y graves", "en": "There are mild and severe patients"}),
            ("correct", {"es": "Se usaron unidades distintas", "en": "Different units were used"}),
            ("dont_know", {"es": "No lo sé", "en": "I don't know"}),
        ],
    },
}

PROFILE_INSTRUCTIONS = {
    "ml": {
        1: ("The user has little machine-learning knowledge. Avoid jargon, define any metric "
            "you mention in one sentence, and use clinical analogies."),
        3: ("The user is comfortable with machine learning. Be concise and technical; you may "
            "discuss metrics, validation strategy and leakage directly."),
    },
    "domain": {
        1: ("The user has little clinical knowledge. When a variable matters, briefly explain "
            "what it measures and when it is recorded in the care process."),
        3: ("The user has clinical expertise. Do not explain clinical concepts; relate results "
            "to clinical practice instead."),
    },
}


def _lang(lang):
    return lang if lang in ("es", "en") else "en"


def format_number(value, lang, digits=3):
    """Round to significant digits and use the decimal comma in Spanish."""
    if not isinstance(value, (int, float)):
        return value
    text = f"{value:.{digits}g}" if abs(value) < 10 ** digits else f"{value:.0f}"
    return text.replace(".", ",") if lang == "es" else text


def format_facts(facts, column=None, description=None, moment=None, lang="en"):
    """Prepare the placeholders shared by all templates."""
    lang = _lang(lang)
    values = {k: format_number(v, lang) for k, v in facts.items()}
    values["column"] = column or ""
    values["description"] = description or ""
    values["moment"] = (moment or "").lower()
    for source, target in (("minority_share", "minority_pct"), ("baseline_accuracy", "baseline_pct"),
                           ("accuracy", "accuracy_pct")):
        if facts.get(source) is not None:
            values[target] = format_number(round(100 * facts[source], 1), lang)
    auc = facts.get("single_feature_auc")
    values["auc_clause"] = f" (AUC {format_number(auc, lang, 2)})" if isinstance(auc, (int, float)) else ""
    return values


def render_text(kind, level, lang, values, stage):
    lang = _lang(lang)
    text = TEXTS[kind][level][lang].format(**values)
    suffix = POST_TRAIN_SUFFIX.get(kind, {}).get(lang)
    if stage == "post_train" and suffix:
        try:
            text += suffix.format(**values)
        except KeyError:
            pass
    return text


def render_title(kind, lang):
    return TITLES[kind][_lang(lang)]


def render_probe(kind, lang, values):
    lang = _lang(lang)
    probe = PROBES[kind]
    return {
        "question": probe["question"][lang].format(**values),
        "options": [{"key": key, "label": label[lang].format(**values)}
                    for key, label in probe["options"]],
    }


def profile_instruction(estimates, min_confidence=0.2):
    """System-prompt addition describing the inferred competence profile."""
    parts = []
    for dim, est in estimates.items():
        if est["confidence"] < min_confidence:
            continue
        if est["score"] < 1.7:
            parts.append(PROFILE_INSTRUCTIONS[dim][1])
        elif est["score"] > 2.4:
            parts.append(PROFILE_INSTRUCTIONS[dim][3])
    return " ".join(parts)
