# Rúbrica para codificar el texto libre de `task_submit.issues_text`

> La codifican **dos evaluadores ciegos a la condición y al perfil**. Antes de codificar se eliminan la condición y el identificador del participante. Para cada trampa se asigna 1 (mencionada) o 0. Se calcula el κ de Cohen por trampa y los desacuerdos se resuelven por consenso.

| Trampa | Codificar 1 si el texto menciona… | Ejemplos que puntúan | Ejemplos que **no** puntúan |
|---|---|---|---|
| M1 desbalance | Pocos positivos, *accuracy* engañosa, comparación con la tasa base, haber usado AUC o F1 por ese motivo | «la mayoría no reingresa, el 88 % lo saca cualquiera» | «la accuracy es alta» (sin crítica) |
| M2 identificador | Que el ID no debe usarse, que es una fuga o que «el número de registro predice» | «quité record_id porque no tiene sentido clínico y salía el primero» | «quité columnas irrelevantes» (sin nombrarla) |
| C1 información posterior | Que la variable se conoce después del momento de la predicción, o que es consecuencia del desenlace o de la sospecha | «el antibiótico se pone cuando ya piensas en sepsis» | «el antibiótico es importante» |
| C2 unidades | Unidades distintas, dos escalas o valores incompatibles en esa variable | «la creatinina está en mg/dL y en µmol/L» | «hay valores atípicos en creatinina» |

Además se registra si el participante menciona **problemas falsos** (falsas alarmas), para analizar la sobrecorrección inducida por la guía.
