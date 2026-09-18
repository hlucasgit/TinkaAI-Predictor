# TinkaAI Predictor - Estado metodologico

## Conclusion operativa actual

La señal operativa seleccionada por la optimizacion cronologica es la frecuencia
de los ultimos 100 sorteos. Los pesos congelados actuales son:

- TimesFM: 0.00
- Frecuencia historica: 0.00
- Frecuencia reciente: 1.00
- Recencia: 0.00

Esto no implica que la frecuencia reciente tenga una ventaja demostrada sobre
una loteria justa. En el holdout externo de 27 sorteos el rendimiento estuvo
cerca del esperado aleatorio y no produjo evidencia estadistica fuerte.

## TimesFM

TimesFM se mantiene como linea experimental, no como componente operativo.

El diagnostico sobre 127 evaluaciones mostro:

- overlap medio Top-6 TimesFM vs sorteo anterior: 3.9528
- repeticion real media entre sorteos consecutivos: 0.7638
- 69.29% de las predicciones TimesFM contenian al menos 4 numeros del sorteo anterior
- score TimesFM medio de numeros del sorteo anterior: 0.8405
- score TimesFM medio del resto: 0.4565

La correccion de persistencia lag-1 empeoro la validacion:

- TimesFM original: 0.7969
- TimesFM debiased: 0.6094
- frecuencia reciente: 0.7969
- baseline lag-1: 0.8281

Por tanto, la señal util de TimesFM en este problema parece estar fuertemente
asociada a persistencia del ultimo sorteo. No se usa como predictor operativo.

## Evaluacion prospectiva

A partir del snapshot congelado posterior al 2026-09-16, los nuevos sorteos deben
evaluarse sin reajustar previamente pesos, ventanas ni reglas.

Archivo de snapshot:

data/processed/prediccion_congelada_actual.json

Evaluador:

python src/models/evaluate_frozen_prediction.py --numbers N1 N2 N3 N4 N5 N6

o, si el sorteo ya esta en tinka_actualizacion_2026.csv:

python src/models/evaluate_frozen_prediction.py --date YYYY-MM-DD

Los resultados prospectivos se acumulan en:

data/processed/evaluaciones_prospectivas.csv
