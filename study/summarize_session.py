"""Readable summary of study sessions, plus a per-trap CSV for analysis.

Usage:
    python study/summarize_session.py                 # list all sessions
    python study/summarize_session.py TEST1           # summarise every session of a participant
    python study/summarize_session.py study_logs/TEST1_ab12cd34ef56.jsonl
    python study/summarize_session.py --csv traps.csv # one row per session x trap

Behavioural detection only: a trap counts as fixed when the final configuration
corrects it, and as self-initiated when the first fix happened before the
system showed anything about that trap. Declared detection (the free text of
the end-of-task form) must still be coded by two raters with
study/instruments/rubrica_codificacion.md.
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(STUDY_DIR)
LOG_DIR = os.getenv("STUDY_LOG_DIR", os.path.join(ROOT, "study_logs"))
DATASET_KEYS = {"A": "hf_readmission", "B": "ed_sepsis"}
ISSUE_KEY = {"M1": "imbalance_metric", "M2": "identifier", "C1": "temporal_leak", "C2": "unit_mixing"}


def load_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def event_type(record):
    return record.get("type") or record.get("client", {}).get("type")


def event_ts(record):
    client = record.get("client", {})
    return client.get("ts") or record.get("server_ts")


def load_traps(dataset_key):
    name = DATASET_KEYS.get(dataset_key)
    path = os.path.join(STUDY_DIR, "answer_key", f"{name}_traps.json")
    if not name or not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh)["traps"]


def analyse(path):
    records = load_records(path)
    created = next((r for r in records if r.get("type") == "session_created"), {})
    events = [r["client"] for r in records if "client" in r]
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    submit = (by_type.get("task_submit") or [None])[-1]
    trainings = by_type.get("train_start", [])
    final_state = (submit or {}).get("final_state") or (trainings[-1]["config"] if trainings else {})
    dropped = set(final_state.get("dropped") or [])
    metric = (final_state.get("metric") or "accuracy").lower()

    first_shown = {}
    for e in by_type.get("guidance_shown", []):
        first_shown.setdefault(e["key"], e["ts"])

    traps = []
    for trap in load_traps(created.get("dataset_key")):
        col = trap["column"]
        if trap["id"] == "M1":
            key = ISSUE_KEY["M1"]
            fixes = [e for e in by_type.get("metric_change", []) if e.get("metric") != "accuracy"]
            fixed = metric != "accuracy"
        else:
            key = f"{ISSUE_KEY[trap['id']]}:{col}"
            fixes = [e for e in by_type.get("column_drop_toggle", [])
                     if e.get("column") == col and e.get("dropped")]
            fixed = col in dropped
        first_fix = fixes[0]["ts"] if fixes else None
        shown = first_shown.get(key)
        traps.append({
            "trap": trap["id"], "dimension": trap["dimension"], "column": col,
            "fixed_final": fixed,
            "self_initiated": bool(first_fix and (shown is None or first_fix < shown)),
            "guidance_shown": shown is not None,
        })

    snapshots = [r for r in records if r.get("type") == "competence_snapshot"]
    start = event_ts(records[0]) if records else None
    end = event_ts(records[-1]) if records else None
    return {
        "path": path,
        "participant": created.get("participant_id"),
        "session": created.get("session_id"),
        "condition": created.get("condition"),
        "dataset": created.get("dataset_key"),
        "started": datetime.fromtimestamp(start / 1000).strftime("%Y-%m-%d %H:%M") if start else "",
        "minutes": round((end - start) / 60000, 1) if start and end else 0,
        "dataset_loaded": bool(by_type.get("dataset_loaded")),
        "trainings": len(trainings),
        "overrides": len(by_type.get("train_override", [])),
        "probes": [(e["key"], e["answer"]) for e in by_type.get("guidance_answer", [])],
        "interventions": sorted({(e["key"], e["mode"], e.get("level")) for e in by_type.get("guidance_shown", [])}),
        "competence": snapshots[-1]["heuristic"] if snapshots else None,
        "submit": submit,
        "traps": traps,
    }


def print_summary(s):
    print(f"\n=== {s['participant']} · sesión {s['session']} · {s['started']} ({s['minutes']} min)")
    print(f"Condición: {s['condition']} · dataset: {s['dataset']} · fichero: {os.path.basename(s['path'])}")
    if not s["dataset_loaded"]:
        print("  (sesión vacía: no se cargó ningún dataset; probablemente una recarga de la página)")
        return
    print(f"Entrenamientos: {s['trainings']} · 'Entrenar igualmente': {s['overrides']}")
    print("Trampas (detección conductual):")
    for t in s["traps"]:
        state = "corregida" if t["fixed_final"] else "NO corregida"
        how = "por iniciativa propia" if t["self_initiated"] else ("tras aviso" if t["fixed_final"] else "")
        shown = "aviso mostrado" if t["guidance_shown"] else "sin aviso"
        print(f"  {t['trap']} ({t['dimension']:6}) {t['column']:20} {state:13} {how:22} [{shown}]")
    if s["probes"]:
        print("Respuestas a preguntas del sistema:")
        for key, answer in s["probes"]:
            print(f"  {key}: {answer}")
    if s["interventions"]:
        print("Avisos mostrados (clave, modo, nivel):")
        for key, mode, level in s["interventions"]:
            print(f"  {key}: {mode}{' / ' + level if level else ''}")
    if s["competence"]:
        c = s["competence"]
        print(f"Perfil inferido final: ML {c['ml']['score']} (confianza {c['ml']['confidence']}) · "
              f"dominio {c['domain']['score']} (confianza {c['domain']['confidence']})")
    if s["submit"]:
        sub = s["submit"]
        print(f"Formulario: desplegaría={'sí' if sub.get('deploy') else 'no'} · confianza={sub.get('confidence')}/7")
        print(f"  Problemas detectados: {sub.get('issues_text') or '(vacío)'}")
    else:
        print("Formulario: no enviado")


def resolve(targets):
    if not targets:
        return sorted(glob.glob(os.path.join(LOG_DIR, "*.jsonl")), key=os.path.getmtime)
    paths = []
    for t in targets:
        if os.path.isfile(t):
            paths.append(t)
        else:
            paths.extend(sorted(glob.glob(os.path.join(LOG_DIR, f"{t}_*.jsonl")), key=os.path.getmtime))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help="participant codes or .jsonl files")
    parser.add_argument("--csv", help="write one row per session x trap to this file")
    parser.add_argument("--all", action="store_true", help="include empty sessions")
    args = parser.parse_args()

    paths = resolve(args.targets)
    if not paths:
        print(f"No hay sesiones en {LOG_DIR}")
        return
    summaries = [analyse(p) for p in paths]
    if not args.all:
        summaries = [s for s in summaries if s["dataset_loaded"]] or summaries
    for s in summaries:
        print_summary(s)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["participant", "session", "condition", "dataset", "trap", "dimension",
                             "column", "fixed_final", "self_initiated", "guidance_shown",
                             "deploy", "confidence", "issues_text"])
            for s in summaries:
                sub = s["submit"] or {}
                for t in s["traps"]:
                    writer.writerow([s["participant"], s["session"], s["condition"], s["dataset"],
                                     t["trap"], t["dimension"], t["column"], int(t["fixed_final"]),
                                     int(t["self_initiated"]), int(t["guidance_shown"]),
                                     sub.get("deploy"), sub.get("confidence"), sub.get("issues_text")])
        print(f"\nCSV escrito en {args.csv}")


if __name__ == "__main__":
    main()
