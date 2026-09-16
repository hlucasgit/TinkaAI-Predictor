"""
Carga del histórico Tinka desde Excel.

Responsabilidades:
- Leer archivo xlsx
- Detectar columnas de números
- Normalizar estructura temporal
- Preparar dataframe para análisis y modelos IA
"""

from pathlib import Path
import pandas as pd


def cargar_excel(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe archivo: {ruta}")

    return pd.read_excel(ruta)


def detectar_columnas_numeros(df: pd.DataFrame):
    columnas = []

    for columna in df.columns:
        nombre = str(columna).lower()
        if any(x in nombre for x in ["num", "bola", "bolilla"]):
            columnas.append(columna)

    if len(columnas) < 6:
        columnas = list(df.columns[-6:])

    return columnas[:6]


def detectar_columna_fecha(df: pd.DataFrame):
    for columna in df.columns:
        nombre = str(columna).lower()
        if "fecha" in nombre or "date" in nombre:
            return columna
    return None


def convertir_formato_largo(df: pd.DataFrame):
    columnas_numero = detectar_columnas_numeros(df)
    columna_fecha = detectar_columna_fecha(df)

    registros = []

    for indice, fila in df.iterrows():
        fecha = fila[columna_fecha] if columna_fecha else indice + 1

        for columna in columnas_numero:
            valor = fila[columna]

            if pd.notna(valor):
                registros.append({
                    "IdSorteo": indice + 1,
                    "Fecha": fecha,
                    "Numero": int(valor),
                    "Aparecio": 1
                })

    resultado = pd.DataFrame(registros)

    if "Fecha" in resultado.columns:
        resultado["Fecha"] = pd.to_datetime(resultado["Fecha"], errors="coerce")

    return resultado


def cargar_excel_tinka(ruta: str | Path) -> pd.DataFrame:
    """Carga y normaliza el histórico para el pipeline principal."""
    df = cargar_excel(ruta)
    return convertir_formato_largo(df)
