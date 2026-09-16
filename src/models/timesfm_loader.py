"""
Carga y preparación inicial para TimesFM.

Este módulo no instala ni ejecuta TimesFM todavía.
Prepara los datos generados por el pipeline para el modelo.
"""

from pathlib import Path
import pandas as pd


def cargar_timesfm_dataset(ruta):
    """Carga timesfm_input.csv."""
    df = pd.read_csv(ruta)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha")
    return df


def separar_contexto_prueba(df, porcentaje_prueba=0.15):
    """Separa histórico y periodo de evaluación."""
    limite = int(len(df) * (1 - porcentaje_prueba))

    contexto = df.iloc[:limite].copy()
    prueba = df.iloc[limite:].copy()

    return contexto, prueba
