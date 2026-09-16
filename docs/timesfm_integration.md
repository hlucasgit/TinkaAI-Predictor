# Integracion TimesFM

## Objetivo

Preparar el proyecto para evaluar TimesFM como modelo de series temporales.

## Flujo

```
Historico Tinka
      |
      v
Normalizacion
      |
      v
Serie temporal
      |
      v
TimesFM
      |
      v
Forecast
```

## Consideraciones

TimesFM no predice resultados de sorteos. Se utilizara para estudiar patrones temporales sobre variables derivadas como frecuencia, aparicion y comportamiento historico.

## Proceso de evaluacion

1. Crear series temporales por numero.
2. Entrenar/evaluar forecast.
3. Comparar contra modelos estadisticos base.
4. Medir error.
