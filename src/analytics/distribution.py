"""
Analisis de distribuciones estadisticas.
"""

import pandas as pd


NUMBER_COLUMNS = [
    "Numero_1",
    "Numero_2",
    "Numero_3",
    "Numero_4",
    "Numero_5",
    "Numero_6",
]


def resumen_sorteos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera indicadores generales de cada sorteo.
    """

    resultado = df.copy()

    resultado["Suma_Numeros"] = resultado[NUMBER_COLUMNS].sum(axis=1)
    resultado["Cantidad_Pares"] = (
        resultado[NUMBER_COLUMNS].apply(lambda x: x % 2 == 0).sum(axis=1)
    )
    resultado["Cantidad_Impares"] = 6 - resultado["Cantidad_Pares"]

    return resultado
