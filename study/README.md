# Estudio: guía adaptativa según la competencia en AutoML clínico

**Hipótesis central:** los no expertos en AutoML clínico son de **dos tipos**.
- Los clínicos dominan el problema pero no el ML.
- Los expertos en IA dominan el ML pero no el problema.

Cada perfil comete errores distintos. La app infiere de forma implícita, a partir de cómo se usa la interfaz, un perfil de competencia en **dos dimensiones** (ML × dominio). Con él decide cuándo y cómo intervenir para evitar los errores de cada perfil, sin interrumpir donde el usuario ya es competente.

- Diseño de las trampas y cómo se puntúan: [`TRAPS.md`](TRAPS.md)
- Instrumentos de medida: [`instruments/`](instruments/)
- Clave de respuestas, solo para investigadores: [`answer_key/`](answer_key/)

## Componentes en el código

| Fichero | Qué hace |
|---|---|
| `static/study.js` | Modo estudio: telemetría, tarea y diccionario, avisos, modal antes de entrenar, formulario final |
| `study_routes.py` | Rutas `/study/*`: sesión, telemetría, perfil del dataset, guía, respuestas a sondas |
| `utils/guardrails.py` | Detectores genéricos de las 4 clases de problema (no leen la clave de respuestas) |
| `utils/competence.py` | Modelo de competencia implícito en 2 dimensiones (bayesiano y transparente) y, si hay modelo local, estimación con la capa de decisión |
| `utils/typed_decision.py` | Capa de decisión tipada al estilo Jev (Choice / Score / Noul) sobre un LLM local con *logprobs* |
| `utils/study.py` | Sesión, política de intervención por condición y registro JSONL |
| `utils/study_messages.py` | Textos por nivel (novato / intermedio / experto), sondas e instrucciones de perfil para el chat |

## Cómo lanzar una sesión

```bash
python study/generate_datasets.py --check   # regenera los datasets (deterministas)
python app.py                               # arranca la app
```

El participante abre esta URL (el investigador la prepara de antemano):

```
http://localhost:5000/?study=1&pid=P07&cond=adaptive&ds=A&lang=es
```

| Parámetro | Valores |
|---|---|
| `pid` | Código seudónimo del participante (no uses nombres) |
| `cond` | `baseline`, `generic` o `adaptive` |
| `ds` | `A` (insuficiencia cardiaca) o `B` (sepsis) |
| `lang` | `es` o `en` |

Sin `?study=1` la app funciona exactamente igual que antes.

### Condiciones

- **`baseline`**: la app publicada. No muestra avisos, pero registra la telemetría y la competencia inferida.
- **`generic`**: todos los problemas se muestran a todo el mundo con el mismo texto (nivel intermedio) y bloquean el entrenamiento.
- **`adaptive`**: la política consciente de la competencia, para la dimensión que corresponde a cada problema:
  - **Confianza < 0,2:** hace una **sonda**, una pregunta concreta cuya respuesta actualiza el perfil.
  - **Nivel < 1,7:** explicación para novato, y bloquea antes de entrenar.
  - **Nivel entre 1,7 y 2,4:** explicación intermedia, sin bloquear.
  - **Nivel > 2,4:** aviso discreto de una línea («Ver detalle»).

  Además, el asistente de chat recibe el perfil inferido.

El control del estudio principal debe ser `generic` o `baseline`; la decisión está en la propuesta. Ambos están implementados.

### Contrabalanceo (2 condiciones × 2 datasets)

| Orden | Tarea 1 | Tarea 2 |
|---|---|---|
| 1 | adaptive · A | control · B |
| 2 | adaptive · B | control · A |
| 3 | control · A | adaptive · B |
| 4 | control · B | adaptive · A |

Asigna los órdenes de forma rotatoria **dentro de cada perfil** (clínico / IA / mixto), para que cada perfil quede repartido de forma equilibrada entre los órdenes.

### Guion de la sesión (unos 75 minutos)

1. Consentimiento informado y [autoevaluación](instruments/autoevaluacion.md) (5 min).
2. Vídeo de 3 minutos sobre la app, igual para todos.
3. Tarea 1 (25 min como máximo). Termina con «Finalizar tarea».
4. Tarea 2 (25 min como máximo).
5. NASA-TLX y SUS tras cada tarea. Tests objetivos de [ML](instruments/test_objetivo_ml.md) y de [dominio](instruments/test_objetivo_dominio.md) al final (10 min).
6. Entrevista breve (5 min).

Para que dé tiempo, fija el presupuesto de entrenamiento a 2–3 minutos.

## Modelo local para la capa de decisión (opcional)

Sin configuración, la competencia se estima con el modelo bayesiano transparente y la fuga semántica con palabras clave. Para usar un LLM local como capa de decisión al estilo Jev:

```bash
# ejemplo con llama.cpp: llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080
export DECISION_BASE_URL=http://localhost:8080
export DECISION_MODEL=qwen2.5-7b-instruct
export COMPETENCE_SOURCE=decision_layer   # o "heuristic" (por defecto)
```

Las dos estimaciones (heurística y de la capa de decisión) se registran **siempre**, así se pueden comparar en el artículo contra los tests objetivos. `COMPETENCE_SOURCE` solo decide cuál usa la política durante la sesión.

## Datos registrados

Cada sesión genera `study_logs/<pid>_<session>.jsonl` (excluido de git). Contiene:
- todos los eventos de interfaz con marca de tiempo;
- el perfil del dataset;
- las estimaciones de competencia tras cada lote de eventos;
- cada decisión de la política, con los problemas detectados y las intervenciones mostradas;
- las respuestas a las sondas;
- el formulario final.

Los mensajes del chat se guardan en texto. Debe constar en el consentimiento informado.

## Tests

```bash
pip install pytest flask pandas numpy
python -m pytest tests
```
