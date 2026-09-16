"""
Generador de dataset temporal para TimesFM.

Convierte sorteos normalizados a una serie temporal multivariada:
Fecha,N01,N02,...,N53
"""

from pathlib import Path
import pandas as pd


def build_timesfm_dataset(input_file, output_file):
    df = pd.read_csv(input_file)

    if "Fecha" not in df.columns:
        raise ValueError("El dataset requiere la columna Fecha")

    if "Numero" not in df.columns:
        raise ValueError("El dataset requiere la columna Numero")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Numero"] = df["Numero"].astype(int)

    df["Valor"] = 1

    matriz = (
        df.pivot_table(
            index="Fecha",
            columns="Numero",
            values="Valor",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    for numero in range(1, 54):
        if numero not in matriz.columns:
            matriz[numero] = 0

    columnas = ["Fecha"] + list(range(1, 54))
    matriz = matriz[columnas]

    matriz.columns = ["Fecha"] + [f"N{x:02d}" for x in range(1,54)]

    matriz.to_csv(output_file, index=False)

    return matriz
