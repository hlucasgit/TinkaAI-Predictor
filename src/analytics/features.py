"""
Generacion de variables analiticas para modelos predictivos.

Este modulo prepara caracteristicas que posteriormente pueden ser usadas
por modelos estadisticos, Machine Learning y TimesFM.
"""

import pandas as pd


def agregar_caracteristicas_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables derivadas de fecha."""

    resultado = df.copy()
    resultado["Fecha_sorteo"] = pd.to_datetime(resultado["Fecha_sorteo"])

    resultado["Anio"] = resultado["Fecha_sorteo"].dt.year
    resultado["Mes"] = resultado["Fecha_sorteo"].dt.month
    resultado["DiaSemana"] = resultado["Fecha_sorteo"].dt.dayofweek

    return resultado


def crear_frecuencia_movil(df: pd.DataFrame, ventana: int = 30) -> pd.DataFrame:
    """
    Calcula frecuencia movil de aparicion de numeros.
    """

    largo = df.copy()
    largo = largo.sort_values("Fecha_sorteo")

    largo["Frecuencia_Movil"] = (
        largo.groupby("Numero")["Aparecio"]
        .rolling(ventana, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    return largo
