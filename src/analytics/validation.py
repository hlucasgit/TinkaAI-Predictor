"""
Validaciones de calidad para datos historicos de Tinka.

La Tinka utiliza un universo de 53 bolillas.
"""

import pandas as pd

NUMERO_MINIMO = 1
NUMERO_MAXIMO = 53

COLUMNAS_NUMEROS = [
    "Numero_1",
    "Numero_2",
    "Numero_3",
    "Numero_4",
    "Numero_5",
    "Numero_6",
]


def validar_rango_numeros(df: pd.DataFrame):
    """Detecta numeros fuera del rango permitido."""
    errores = []

    for columna in COLUMNAS_NUMEROS:
        if columna not in df.columns:
            continue

        invalidos = df[
            (df[columna] < NUMERO_MINIMO)
            | (df[columna] > NUMERO_MAXIMO)
        ]

        for _, fila in invalidos.iterrows():
            errores.append(
                {
                    "tipo": "NUMERO_FUERA_RANGO",
                    "columna": columna,
                    "valor": fila[columna],
                    "fecha": fila.get("Fecha_sorteo"),
                }
            )

    return pd.DataFrame(errores)


def validar_duplicados_sorteo(df: pd.DataFrame):
    """Detecta numeros repetidos dentro de un mismo sorteo."""
    errores = []

    for _, fila in df.iterrows():
        numeros = [fila[c] for c in COLUMNAS_NUMEROS if c in df.columns]
        if len(numeros) != len(set(numeros)):
            errores.append(
                {
                    "tipo": "NUMERO_DUPLICADO",
                    "fecha": fila.get("Fecha_sorteo"),
                    "numeros": numeros,
                }
            )

    return pd.DataFrame(errores)


def reporte_calidad(df: pd.DataFrame):
    """Genera un resumen general de calidad."""
    return {
        "registros": len(df),
        "columnas": list(df.columns),
        "numeros_validos": f"{NUMERO_MINIMO}-{NUMERO_MAXIMO}",
        "duplicados": len(validar_duplicados_sorteo(df)),
        "fuera_rango": len(validar_rango_numeros(df)),
    }
