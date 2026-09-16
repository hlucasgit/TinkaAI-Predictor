"""
Analisis de frecuencia de numeros de sorteos.
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


def transformar_a_formato_largo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un sorteo horizontal a formato analitico largo.

    Entrada:
        Fecha_sorteo | Numero_1 ... Numero_6

    Salida:
        Fecha_sorteo | Numero | Aparecio
    """

    registros = []

    for _, row in df.iterrows():
        for columna in NUMBER_COLUMNS:
            registros.append(
                {
                    "Fecha_sorteo": row["Fecha_sorteo"],
                    "Numero": int(row[columna]),
                    "Aparecio": 1,
                }
            )

    return pd.DataFrame(registros)


def frecuencia_numeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene cantidad de apariciones por numero.
    """

    largo = transformar_a_formato_largo(df)

    return (
        largo.groupby("Numero")
        .size()
        .reset_index(name="Cantidad_Apariciones")
        .sort_values("Cantidad_Apariciones", ascending=False)
    )
