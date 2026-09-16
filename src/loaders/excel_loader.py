"""
Carga del histórico Tinka desde Excel.

Responsabilidades:
- Leer archivo xlsx
- Detectar columnas de números
- Normalizar estructura
- Preparar dataframe para análisis
"""

from pathlib import Path
import pandas as pd


NUMERO_MINIMO = 1
NUMERO_MAXIMO = 53


def cargar_excel(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe archivo: {ruta}")

    df = pd.read_excel(ruta)

    return df


def detectar_columnas_numeros(df: pd.DataFrame):
    columnas = []

    for columna in df.columns:
        nombre = str(columna).lower()
        if any(x in nombre for x in ["num", "bola", "bolilla"]):
            columnas.append(columna)

    if len(columnas) < 6:
        columnas = list(df.columns[-6:])

    return columnas[:6]


def convertir_formato_largo(df: pd.DataFrame):
    columnas_numero = detectar_columnas_numeros(df)

    registros = []

    for _, fila in df.iterrows():
        for columna in columnas_numero:
            valor = fila[columna]

            if pd.notna(valor):
                registros.append({
                    "Numero": int(valor),
                    "Aparecio": 1
                })

    return pd.DataFrame(registros)
