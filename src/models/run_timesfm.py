"""
Ejecutor inicial TimesFM para TinkaAI Predictor.

Carga el dataset temporal generado y prepara la inferencia
con TimesFM 2.5 backend PyTorch.
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
    df = df.sort_values("Fecha")
    return df


def preparar_contexto(df):
    columnas = [c for c in df.columns if c.startswith("N")]
    return df[columnas].values.astype(np.float32), columnas


def main():
    print("=" * 60)
    print("TinkaAI Predictor - TimesFM")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = cargar_serie()
    contexto, columnas = preparar_contexto(df)

    print(f"Registros temporales: {len(df)}")
    print(f"Variables: {len(columnas)}")

    try:
        from timesfm.timesfm_2p5 import timesfm_2p5_torch
        print("TimesFM PyTorch disponible")
        print(timesfm_2p5_torch.TimesFM_2p5_200M_torch)
    except Exception as e:
        raise RuntimeError(f"Error cargando TimesFM: {e}")

    print("Contexto preparado correctamente")
    print("Siguiente paso: cargar checkpoint y ejecutar forecast")


if __name__ == "__main__":
    main()
