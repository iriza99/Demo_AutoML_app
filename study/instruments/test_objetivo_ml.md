# Test objetivo de machine learning (al final de la sesión)

> Se administra **después** de ambas tareas para no condicionar el comportamiento. Tiene 8 ítems de opción múltiple: 1 punto por acierto, sin penalización. «No lo sé» cuenta como error.
> **Niveles:** 0–3 → 1 (bajo), 4–6 → 2 (medio), 7–8 → 3 (alto). Hay que validar los puntos de corte en el piloto.

1. Un test diagnóstico para una enfermedad con prevalencia del 2 % tiene una *accuracy* del 98 %. ¿Qué puedes concluir?
   a) Es un test excelente · **b) No se puede concluir nada: un test que siempre diga «sano» obtiene lo mismo** · c) Tiene una sensibilidad del 98 % · d) No lo sé
2. ¿Qué mide el área bajo la curva ROC (AUC)?
   a) El porcentaje de aciertos · **b) La probabilidad de que el modelo puntúe más alto a un positivo que a un negativo elegidos al azar** · c) La calibración del modelo · d) No lo sé
3. Un modelo obtiene un AUC de 0,99 en validación cruzada pero falla en producción. ¿Cuál es la causa **más probable**?
   a) Pocos árboles · **b) Fuga de información (*data leakage*)** · c) Falta de normalización · d) No lo sé
4. ¿Para qué sirve la validación cruzada?
   a) Para aumentar el tamaño del dataset · **b) Para estimar el rendimiento en datos no vistos usando todos los datos** · c) Para seleccionar las variables · d) No lo sé
5. En un gráfico de importancia de variables, el identificador del paciente aparece como la variable más importante. ¿Qué indica?
   a) Que el identificador contiene información clínica · **b) Que hay un artefacto o fuga en cómo se construyeron los datos** · c) Que el modelo está bien calibrado · d) No lo sé
6. ¿Qué es el sobreajuste (*overfitting*)?
   a) Que el modelo tarda mucho en entrenar · **b) Que el modelo memoriza el ruido del entrenamiento y generaliza mal** · c) Que hay demasiadas variables categóricas · d) No lo sé
7. Con un desenlace raro (10 %), ¿qué métrica es **menos** informativa por sí sola?
   **a) *Accuracy*** · b) AUC · c) F1 · d) No lo sé
8. Un valor SHAP positivo para una variable en un paciente significa que…
   a) La variable es significativa estadísticamente · **b) Esa variable empuja la predicción de ese paciente hacia la clase positiva** · c) La variable tiene un valor alto · d) No lo sé
