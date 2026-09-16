"""
Evaluación histórica del modelo TinkaAI.

Permite comparar predicciones contra sorteos conocidos.
"""

from pathlib import Path
import pandas as pd


def split_temporal(df, porcentaje_entrenamiento=0.8):
    df = df.sort_values("Fecha").reset_index(drop=True)
    limite = int(len(df) * porcentaje_entrenamiento)
    return df.iloc[:limite], df.iloc[limite:]


def calcular_metricas(prediccion, real):
    resultados = {}

    for columna in real.columns:
        if columna == "Fecha":
            continue

        resultados[columna] = {
            "aciertos": int(((prediccion[columna] >= 0.5) == (real[columna] == 1)).sum()),
            "total": len(real)
        }

    return pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "Numero"})


def guardar_evaluacion(df, archivo_salida):
    Path(archivo_salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(archivo_salida, index=False)
