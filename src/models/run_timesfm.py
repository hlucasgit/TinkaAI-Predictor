"""
Ejecutor TimesFM para TinkaAI Predictor.

Carga el dataset temporal Tinka y ejecuta forecast con TimesFM 2.5 PyTorch.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import timesfm


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "prediccion_timesfm.csv"


def cargar_serie():
    df = pd.read_csv(INPUT_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha")


def preparar_series(df):
    columnas = [c for c in df.columns if c.startswith("N")]

    # TimesFM trabaja con una lista de series univariadas.
    series = [
        df[col].values.astype(np.float32)
        for col in columnas
    ]

    return series, columnas


def cargar_modelo():
    modelo = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    modelo.compile(
        timesfm.ForecastConfig(
            max_context=2048,
            max_horizon=16,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    return modelo


def ejecutar_forecast(modelo, series):
    point, quantiles = modelo.forecast(
        horizon=16,
        inputs=series,
    )

    return point, quantiles


def main():
    print("=" * 60)
    print("TinkaAI Predictor - TimesFM Forecast")
    print("=" * 60)

    df = cargar_serie()
    series, columnas = preparar_series(df)

    print(f"Registros temporales: {len(df)}")
    print(f"Series preparadas: {len(series)}")

    print("Cargando checkpoint TimesFM...")
    modelo = cargar_modelo()

    print("Modelo cargado")
    print("Ejecutando forecast...")

    point, quantiles = ejecutar_forecast(modelo, series)

    fechas = pd.date_range(
        start=df["Fecha"].max(),
        periods=17,
        freq="W"
    )[1:]

    resultado = pd.DataFrame({"Fecha": fechas})

    for i, nombre in enumerate(columnas):
        resultado[nombre] = point[i]

    resultado.to_csv(OUTPUT_FILE, index=False)

    print(f"Prediccion generada: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
