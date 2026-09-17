"""
Ejecutor TimesFM para TinkaAI Predictor.

Carga el dataset temporal Tinka y ejecuta forecast con TimesFM 2.5 PyTorch.
Cada fila del forecast representa un siguiente sorteo (paso temporal), sin
inventar una fecha calendario cuando la cadencia historica no es uniforme.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import timesfm


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "prediccion_timesfm.csv"
HORIZON = 16


def cargar_serie():
    df = pd.read_csv(INPUT_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha")


def preparar_series(df):
    columnas = [c for c in df.columns if c.startswith("N")]
    if len(columnas) != 53:
        raise ValueError(f"Se esperaban 53 columnas N01..N53 y se encontraron {len(columnas)}")

    # TimesFM recibe una lista de series univariadas; cada Nxx es una serie 0/1.
    series = [df[col].values.astype(np.float32) for col in columnas]
    return series, columnas


def cargar_modelo():
    modelo = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    modelo.compile(
        timesfm.ForecastConfig(
            max_context=2048,
            max_horizon=HORIZON,
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
        horizon=HORIZON,
        inputs=series,
    )
    return point, quantiles


def main():
    print("=" * 60)
    print("TinkaAI Predictor - TimesFM Forecast")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = cargar_serie()
    series, columnas = preparar_series(df)

    print(f"Registros temporales: {len(df)}")
    print(f"Series preparadas: {len(series)}")
    print(f"Ultima fecha historica: {df['Fecha'].max().date()}")

    print("Cargando checkpoint TimesFM...")
    modelo = cargar_modelo()

    print("Modelo cargado")
    print("Ejecutando forecast...")

    point, _ = ejecutar_forecast(modelo, series)

    # No se asignan fechas artificiales: cada paso corresponde al siguiente sorteo.
    resultado = pd.DataFrame({"PasoForecast": range(1, HORIZON + 1)})

    for i, nombre in enumerate(columnas):
        resultado[nombre] = point[i]

    resultado.to_csv(OUTPUT_FILE, index=False)

    print(f"Prediccion generada: {OUTPUT_FILE}")
    print("Cada PasoForecast representa un sorteo futuro, no una fecha calendario estimada.")


if __name__ == "__main__":
    main()
