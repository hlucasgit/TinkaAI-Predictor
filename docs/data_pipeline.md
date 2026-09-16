# Pipeline de datos TinkaAI

## Flujo actual

```
Excel historico
      |
      v
Carga y validacion
      |
      v
Normalizacion de sorteos
      |
      v
Formato analitico largo
      |
      +---- Frecuencias
      |
      +---- Patrones
      |
      +---- Variables para IA
      |
      v
Modelos predictivos
```

## Variables preparadas

- Fecha del sorteo
- Año
- Mes
- Día de semana
- Frecuencia móvil

Estas variables serán utilizadas en la etapa de evaluación de modelos.
