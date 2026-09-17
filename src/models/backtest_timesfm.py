"""Backtest temporal de TimesFM contra sorteos historicos conocidos.

Reserva los ultimos HORIZON sorteos, pronostica desde el historial anterior
y evalua cuantas bolillas reales aparecen dentro del Top-6 por score.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import timesfm

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_timesfm_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_timesfm_resumen.csv"
HORIZON = 16
TOP_K = 6
TOTAL_NUMEROS = 53


def cargar_datos():
    df = pd.read_csv(INPUT_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha").reset_index(drop=True)


def cargar_modelo():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
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
    return model


def top_k_numeros(scores, columnas, k=TOP_K):
    indices = np.argsort(scores)[::-1][:k]
    return sorted(int(columnas[i][1:]) for i in indices)


def main():
    print("=" * 60)
    print("TinkaAI Predictor - Backtest TimesFM")
    print("=" * 60)

    df = cargar_datos()
    columnas = [c for c in df.columns if c.startswith("N")]
    if len(columnas) != TOTAL_NUMEROS:
        raise ValueError(f"Se esperaban {TOTAL_NUMEROS} series y se encontraron {len(columnas)}")
    if len(df) <= HORIZON:
        raise ValueError("No hay suficientes sorteos para realizar el backtest")

    train = df.iloc[:-HORIZON].copy()
    test = df.iloc[-HORIZON:].copy().reset_index(drop=True)

    series = [train[c].values.astype(np.float32) for c in columnas]

    print(f"Sorteos de contexto: {len(train)}")
    print(f"Sorteos reservados para prueba: {len(test)}")
    print(f"Periodo de prueba: {test['Fecha'].min().date()} a {test['Fecha'].max().date()}")
    print("Cargando modelo...")

    model = cargar_modelo()
    point, _ = model.forecast(horizon=HORIZON, inputs=series)

    detalle = []
    for paso in range(HORIZON):
        predichos = top_k_numeros(point[:, paso], columnas)
        reales = sorted(
            int(c[1:]) for c in columnas if int(test.loc[paso, c]) == 1
        )
        aciertos = sorted(set(predichos) & set(reales))

        detalle.append({
            "PasoForecast": paso + 1,
            "FechaReal": test.loc[paso, "Fecha"].date().isoformat(),
            "PrediccionTop6": "-".join(f"{n:02d}" for n in predichos),
            "NumerosReales": "-".join(f"{n:02d}" for n in reales),
            "Aciertos": len(aciertos),
            "NumerosAcertados": "-".join(f"{n:02d}" for n in aciertos),
        })

    detalle_df = pd.DataFrame(detalle)
    detalle_df.to_csv(DETAIL_FILE, index=False)

    promedio = float(detalle_df["Aciertos"].mean())
    esperado_azar = TOP_K * TOP_K / TOTAL_NUMEROS
    resumen = pd.DataFrame([{
        "SorteosEvaluados": HORIZON,
        "AciertosTotales": int(detalle_df["Aciertos"].sum()),
        "AciertosPromedioPorSorteo": promedio,
        "EsperadoAzarTop6": esperado_azar,
        "SorteosCon0Aciertos": int((detalle_df["Aciertos"] == 0).sum()),
        "SorteosCon1OMas": int((detalle_df["Aciertos"] >= 1).sum()),
        "MaximoAciertosEnUnSorteo": int(detalle_df["Aciertos"].max()),
    }])
    resumen.to_csv(SUMMARY_FILE, index=False)

    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"Aciertos promedio Top-6: {promedio:.4f}")
    print(f"Referencia teorica aleatoria Top-6: {esperado_azar:.4f}")
    print("La comparacion con azar es descriptiva; 16 sorteos es una muestra pequena.")


if __name__ == "__main__":
    main()
