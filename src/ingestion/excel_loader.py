"""
Carga y validacion de archivos historicos de la Tinka.

Este modulo prepara los datos para los procesos de analisis estadistico
 y modelos predictivos.
"""

from pathlib import Path
from typing import List

import pandas as pd


COLUMNAS_REQUERIDAS = [
    "Fecha_sorteo",
    "Numero_1",
    "Numero_2",
    "Numero_3",
    "Numero_4",
    "Numero_5",
    "Numero_6",
]


class TinkaExcelLoader:
    """Carga y valida el historico de sorteos."""

    def __init__(self, archivo: str):
        self.archivo = Path(archivo)

    def cargar(self) -> pd.DataFrame:
        if not self.archivo.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.archivo}"
            )

        df = pd.read_excel(self.archivo)
        return self.validar(df)

    def validar(self, df: pd.DataFrame) -> pd.DataFrame:
        faltantes = [
            c for c in COLUMNAS_REQUERIDAS
            if c not in df.columns
        ]

        if faltantes:
            raise ValueError(
                f"Columnas faltantes: {faltantes}"
            )

        df["Fecha_sorteo"] = pd.to_datetime(
            df["Fecha_sorteo"],
            errors="coerce"
        )

        numeros = [
            f"Numero_{i}" for i in range(1, 7)
        ]

        for columna in numeros:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

            if df[columna].isna().any():
                raise ValueError(
                    f"Valores invalidos en {columna}"
                )

            if not df[columna].between(1, 48).all():
                raise ValueError(
                    f"Numeros fuera del rango permitido en {columna}"
                )

        return df.sort_values("Fecha_sorteo").reset_index(drop=True)


if __name__ == "__main__":
    loader = TinkaExcelLoader(
        "data/raw/tinka_dataset.xlsx"
    )

    datos = loader.cargar()
    print(datos.head())
    print(f"Registros cargados: {len(datos)}")
