# Diseño de las trampas del estudio

**Pregunta que deben responder las trampas:** ¿cada perfil de usuario cae en *su* punto ciego, y la guía adaptativa lo evita?

- Las trampas de **ML** exigen conocimiento de *machine learning* y un clínico tiende a no verlas.
- Las trampas **clínicas** exigen conocimiento del dominio y un experto en IA tiende a no verlas.

Cada dataset lleva **cuatro trampas emparejadas**: dos de ML y dos clínicas. Las de un dataset son equivalentes a las del otro, así que da igual qué dataset reciba cada condición. Con 30 participantes × 2 tareas × 4 trampas se obtienen **240 observaciones binarias** para la variable principal (trampa detectada o no).

Los datos son sintéticos y se generan de forma reproducible con [`generate_datasets.py`](generate_datasets.py), con semillas fijas. Ventajas frente a usar datos reales:
- Controlamos la fuerza de cada trampa.
- No hay datos de pacientes, lo que simplifica la aprobación ética.
- Conocemos la verdad de referencia.

Las variables y sus efectos se han elegido para que sean plausibles para un clínico.

## Los dos datasets

| | **A. Reingreso por insuficiencia cardiaca** | **B. Sepsis en urgencias** |
|---|---|---|
| Fichero | `datasets/hf_readmission.csv` | `datasets/ed_sepsis.csv` |
| Variable objetivo | `readmission_30d` (reingreso en 30 días) | `sepsis_48h` (sepsis en 48 h) |
| Momento de la predicción | Alta hospitalaria | Triaje de urgencias |
| Filas / columnas | 1 600 / 20 | 1 600 / 20 |
| Prevalencia | 11,6 % | 11,1 % |
| AUC (regresión logística, CV) **sin** trampas | 0,80 | 0,81 |
| AUC **con** las fugas dentro | 0,95 | 0,98 |

El participante recibe la tarea y un **diccionario de datos**, ambos visibles en la app en modo estudio. El diccionario da la descripción de cada variable y el momento de la predicción. Sin esa información la fuga temporal no se podría detectar, y tenerla es lo realista: un clínico siempre conoce el contexto de sus datos.

## Las cuatro trampas

### M1 · Desbalance de clases con *accuracy* (dimensión ML)

|  | A | B |
|---|---|---|
| Implementación | 11,6 % de positivos | 11,1 % de positivos |
| *Accuracy* del modelo trivial («siempre no») | 88,4 % | 88,9 % |

- **Por qué es una trampa:** la métrica por defecto de la app es *accuracy*. Un modelo que no sirve obtiene casi un 90 % y parece excelente.
- **Quién cae, según la hipótesis:** el perfil clínico sin formación en ML.
- **Detección conductual (log):** cambia la métrica a ROC AUC o F1 antes de entrenar, **sin** que el sistema se lo haya señalado.
- **Detección declarada:** en el formulario final menciona el desbalance, la *accuracy* engañosa o compara con la tasa base.

### M2 · Identificador con fuga de información (dimensión ML)

|  | A | B |
|---|---|---|
| Columna | `record_id` | `episode_id` |
| AUC de esa columna sola | 0,85 | 0,87 |

- **Cómo se ha creado:** los identificadores se asignaron después de ordenar por el desenlace. Simula una exportación en la que los casos positivos se añadieron desde otro registro.
- **Por qué es una trampa:** el número de registro «predice» el desenlace. En pacientes nuevos no funcionaría.
- **Quién cae:** el perfil clínico, que ve un ID como algo inofensivo.
- **Detección conductual:** excluye la columna antes de entrenar, o la excluye y reentrena tras ver que aparece entre las más importantes.
- **Detección declarada:** menciona que el ID no debería usarse, o habla de fuga o de que el modelo «hace trampa».

### C1 · Información posterior al momento de la predicción (dimensión dominio)

| | A | B |
|---|---|---|
| Columna | `nt_probnp_followup` | `antibiotic_iv_ed` |
| Descripción en el diccionario | «NT-proBNP en la revisión ambulatoria» | «Antibiótico IV administrado en urgencias» |
| Por qué es una fuga | La revisión ambulatoria es posterior al alta, y a menudo al propio reingreso | El antibiótico se pone *porque* el clínico ya sospecha sepsis, después del triaje |
| AUC de esa columna sola | 0,91 | 0,90 |

- **Por qué es una trampa:** el modelo usa información del futuro. Las métricas serán espectaculares y el modelo no se podrá usar en la práctica.
- **Quién cae:** el perfil IA, que ve una variable de laboratorio o un tratamiento perfectamente legítimos.
- **Detección conductual:** excluye la columna antes de entrenar, o al ver que domina la importancia.
- **Detección declarada:** dice que la variable no está disponible en el momento de la predicción, o que es una consecuencia del desenlace o de la sospecha clínica.

### C2 · Mezcla de unidades (dimensión dominio)

|  | A | B |
|---|---|---|
| Columna | `creatinine` | `lactate` |
| Unidades mezcladas | mg/dL (~1,2) y µmol/L (~105), factor 88,4 | mmol/L (~1,8) y mg/dL (~16), factor 9,01 |
| Proporción de registros en la unidad alternativa | ~40 % | ~40 % |

- **Cómo se ha creado:** un hospital (oculto, no aparece como columna) registra la variable en otra unidad y además tiene una tasa base de eventos algo distinta.
- **Por qué es una trampa:** el modelo aprende una relación espuria («creatinina alta → riesgo») que en realidad refleja el hospital. Su interpretación clínica es errónea.
- **Quién cae:** el perfil IA. Puede ver la distribución bimodal, pero no la reconoce como un problema de unidades y tiende a tratarla como asimetría o como valores atípicos.
- **Detección conductual:** excluye la columna. La app no tiene conversión de unidades, así que la detección se mide sobre todo por la vía declarada.
- **Detección declarada:** menciona las unidades distintas, o que los valores son incompatibles o están en dos escalas.

## Cómo se puntúa la detección (variable principal)

Para cada participante, tarea y trampa, `detectada = 1` si se cumple **al menos una** de estas condiciones:

1. **Conductual, sin ayuda:** la acción correctiva aparece en el log **antes** de que el sistema haya mostrado una intervención sobre esa trampa (eventos `column_drop_toggle`, `metric_change` y `guidance_shown` en `study_logs/`).
2. **Conductual, tras la intervención:** la acción correctiva aparece **después** de una intervención.
3. **Declarada:** el texto de `task_submit.issues_text` la menciona. Lo codifican **dos evaluadores ciegos a la condición**, con la rúbrica de [`instruments/rubrica_codificacion.md`](instruments/rubrica_codificacion.md). Se reporta el κ de Cohen entre ambos.

Tanto la detección con ayuda (1 o 2) como la detección sin ayuda (solo 1) se analizan por separado, porque significan cosas distintas:
- La detección **con ayuda** mide si la guía consigue modelos válidos (la eficacia del sistema).
- La detección **sin ayuda** solo tiene sentido como medida de competencia en la condición *baseline*, y sirve para validar el modelo de usuario.

**Variables secundarias a nivel de tarea:**
- validez del modelo final (las 3 trampas corregibles M1, M2 y C1 resueltas en la configuración final);
- decisión de desplegar (con fugas, lo correcto es *no*);
- confianza en la decisión (calibración);
- número de interrupciones bloqueantes;
- tiempo hasta enviar la tarea.

## Validez de los detectores

Los detectores de [`utils/guardrails.py`](../utils/guardrails.py) **no leen la clave de respuestas** (`answer_key/`). Encuentran las trampas a partir de los datos y del diccionario:

| Trampa | Detector genérico |
|---|---|
| M1 | Clase minoritaria por debajo del 20 % y métrica `accuracy` |
| M2 | Valores enteros únicos y casi consecutivos, o nombre de identificador |
| C1 | AUC univariante ≥ 0,85 («demasiado bueno»), o la descripción indica que el dato se obtiene después. Esa comprobación la hace la capa de decisión local; sin modelo local, se usa una lista de palabras clave como respaldo |
| C2 | Mezcla de dos gaussianas sobre log(valor), cuya distancia coincide con un factor de conversión clínico conocido (creatinina, lactato, glucosa, colesterol, bilirrubina, urea, hemoglobina…) |

`tests/test_guardrails.py` comprueba que en ambos datasets se detectan las 4 trampas **y nada más**, y que variables naturales asimétricas o normales no producen falsos positivos.

**Pendiente para el artículo:** evaluar los detectores sobre datasets clínicos públicos externos (por ejemplo, de UCI o PhysioNet) para estimar su tasa de falsos positivos fuera de este estudio.

## Riesgos y controles

- **Que alguien detecte las trampas por el nombre de la columna.** Los nombres siguen convenciones reales (`record_id`, `nt_probnp_followup`) y no se han ocultado. Si un perfil clínico detecta C1 por la palabra *followup*, eso **es** conocimiento del dominio.
  - El piloto debe comprobar que ningún perfil detecta el 100 % de las trampas sin ayuda (efecto techo) ni el 0 % (efecto suelo).
- **Aprendizaje entre tareas.** Tras la primera tarea, el participante puede buscar el mismo tipo de trampa en la segunda. Lo controlan el contrabalanceo (orden × dataset) y la inclusión del orden como covariable.
- **Que el chat delate las trampas.** El asistente LLM está disponible en todas las condiciones y ve las estadísticas descriptivas. Si señala trampas por su cuenta, eso afecta a ambas condiciones por igual. Las preguntas y respuestas quedan registradas (`chat_send`).
