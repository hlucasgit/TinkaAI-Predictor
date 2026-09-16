"""
Ejecutor TimesFM para TinkaAI Predictor.

Carga el dataset temporal Tinka y ejecuta una primera inferencia
con TimesFM 2.5 PyTorch.
"""

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "prediccion_timesfm.csv"


def cargar_serie():
    df = pd.read_csv(INPUT_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha")


def preparar_contexto(df):
    columnas = [c for c in df.columns if c.startswith("N")]
    valores = df[columnas].values.astype(np.float32)
    return valores, columnas


def cargar_modelo():
    from timesfm.timesfm_2p5 import timesfm_2p5_torch

    modelo = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    return modelo


def ejecutar_forecast(modelo, contexto):
    """
    Punto de integración con la API forecast de TimesFM.
    Se mantiene separado para adaptar parámetros según la versión instalada.
    """
    return modelo.forecast(contexto)


def main():
    print("=" * 60)
    print("TinkaAI Predictor - TimesFM Forecast")
    print("=" * 60)

    df = cargar_serie()
    contexto, columnas = preparar_contexto(df)

    print(f"Registros temporales: {len(df)}")
    print(f"Variables: {len(columnas)}")

    print("Cargando checkpoint TimesFM...")
    modelo = cargar_modelo()

    print("Modelo cargado correctamente")

    print("Ejecutando forecast...")
    prediccion = ejecutar_forecast(modelo, contexto)

    resultado = pd.DataFrame(prediccion)
    resultado.to_csv(OUTPUT_FILE, index=False)

    print(f"Prediccion generada: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
