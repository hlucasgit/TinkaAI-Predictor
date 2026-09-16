"""
Generador de reportes estadisticos del historico de sorteos.

Este modulo consolida metricas para comparar posteriormente
modelos estadisticos, simulaciones y TimesFM.
"""

import pandas as pd


def resumen_frecuencias(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking de numeros por cantidad de apariciones."""

    return (
        df.groupby("Numero")
        .size()
        .reset_index(name="Apariciones")
        .sort_values("Apariciones", ascending=False)
    )


def distribucion_pares_impares(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula cantidad de numeros pares e impares por sorteo."""

    resultado = []

    columnas = [
        "Numero_1",
        "Numero_2",
        "Numero_3",
        "Numero_4",
        "Numero_5",
        "Numero_6",
    ]

    for _, row in df.iterrows():
        numeros = [row[c] for c in columnas]

        resultado.append(
            {
                "Fecha_sorteo": row["Fecha_sorteo"],
                "Pares": sum(1 for n in numeros if n % 2 == 0),
                "Impares": sum(1 for n in numeros if n % 2 != 0),
                "Suma": sum(numeros),
            }
        )

    return pd.DataFrame(resultado)


def numeros_atrasados(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula sorteos transcurridos desde la ultima aparicion."""

    ultimo_sorteo = df["Fecha_sorteo"].max()
    largo = df.sort_values("Fecha_sorteo")

    ultimo = (
        largo.groupby("Numero")["Fecha_sorteo"]
        .max()
        .reset_index()
    )

    ultimo["Dias_Ausente"] = (
        ultimo_sorteo - ultimo["Fecha_sorteo"]
    ).dt.days

    return ultimo.sort_values("Dias_Ausente", ascending=False)
