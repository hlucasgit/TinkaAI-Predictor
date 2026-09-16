"""
TinkaAI Predictor - Predictor TimesFM

Capa de integración con TimesFM.
La implementación real del modelo se activa cuando
el entorno TimesFM esté instalado.
"""

from pathlib import Path
import pandas as pd


def cargar_dataset_timesfm(path):
    """Carga dataset preparado para TimesFM."""
    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha")


def preparar_contexto(df, columnas=None):
    """Prepara la matriz temporal que será entregada al modelo."""
    if columnas is None:
        columnas = [c for c in df.columns if c.startswith("N")]

    return df[columnas].values


def guardar_prediccion(prediccion, output_file):
    """Guarda resultados futuros generados por TimesFM."""
    resultado = pd.DataFrame(prediccion)
    resultado.to_csv(output_file, index=False)
    return resultado


if __name__ == "__main__":
    print("Módulo TimesFM preparado. Instalar TimesFM para ejecutar predicción.")
