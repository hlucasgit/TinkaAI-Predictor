"""Backtest rolling por bloques para TimesFM en TinkaAI Predictor.

Evalua varios bloques historicos recientes para reducir la dependencia de una
sola ventana de 16 sorteos. Compara TimesFM contra:
1) referencia aleatoria teorica Top-6,
2) baseline de frecuencia reciente.

Cada bloque usa solo informacion disponible antes del periodo evaluado.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_rolling_timesfm_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_rolling_timesfm_resumen.csv"

TOTAL_NUMEROS = 53
TOP_K = 6
MAX_CONTEXT = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest rolling TimesFM sobre sorteos historicos de Tinka"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Cantidad de bloques historicos a evaluar (default: 8)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Sorteos por bloque (default: 16)",
    )
    parser.add_argument(
        "--freq-window",
        type=int,
        default=100,
        help="Ventana de sorteos para baseline de frecuencia (default: 100)",
    )
    return parser.parse_args()


def cargar_datos():
    df = pd.read_csv(INPUT_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha").reset_index(drop=True)


def cargar_modelo(horizon: int):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def top_k_scores(scores, columnas, k=TOP_K):
    indices = np.argsort(np.asarray(scores))[::-1][:k]
    return sorted(int(columnas[i][1:]) for i in indices)


def top_k_frecuencia(train: pd.DataFrame, columnas, ventana: int, k=TOP_K):
    base = train.tail(ventana)
    frecuencias = base[columnas].sum(axis=0).to_numpy(dtype=float)
    return top_k_scores(frecuencias, columnas, k)


def numeros_reales(fila: pd.Series, columnas):
    return sorted(int(c[1:]) for c in columnas if int(fila[c]) == 1)


def aciertos(predichos, reales):
    return sorted(set(predichos) & set(reales))


def formato_numeros(nums):
    return "-".join(f"{n:02d}" for n in nums)


def estadistica_azar(n_sorteos: int):
    # X ~ Hipergeometrica(N=53, K=6, n=6) para un Top-6 aleatorio.
    media = TOP_K * TOP_K / TOTAL_NUMEROS
    varianza = (
        TOP_K
        * (TOP_K / TOTAL_NUMEROS)
        * (1 - TOP_K / TOTAL_NUMEROS)
        * ((TOTAL_NUMEROS - TOP_K) / (TOTAL_NUMEROS - 1))
    )
    error_estandar_media = math.sqrt(varianza / n_sorteos)
    return media, varianza, error_estandar_media


def p_value_normal(z: float):
    # Aproximacion bilateral N(0,1), suficiente como diagnostico descriptivo.
    return math.erfc(abs(z) / math.sqrt(2.0))


def main():
    args = parse_args()
    if args.blocks <= 0 or args.horizon <= 0 or args.freq_window <= 0:
        raise ValueError("blocks, horizon y freq-window deben ser mayores que cero")

    df = cargar_datos()
    columnas = [c for c in df.columns if c.startswith("N")]

    if len(columnas) != TOTAL_NUMEROS:
        raise ValueError(
            f"Se esperaban {TOTAL_NUMEROS} series y se encontraron {len(columnas)}"
        )

    total_test = args.blocks * args.horizon
    if len(df) <= total_test:
        raise ValueError(
            f"No hay suficientes sorteos: se requieren mas de {total_test} registros"
        )

    inicio_test = len(df) - total_test

    print("=" * 68)
    print("TinkaAI Predictor - Backtest Rolling TimesFM")
    print("=" * 68)
    print(f"Sorteos historicos: {len(df)}")
    print(f"Bloques: {args.blocks}")
    print(f"Horizonte por bloque: {args.horizon}")
    print(f"Sorteos evaluados: {total_test}")
    print(
        f"Periodo evaluado: {df.loc[inicio_test, 'Fecha'].date()} "
        f"a {df.iloc[-1]['Fecha'].date()}"
    )
    print("Cargando modelo TimesFM una sola vez...")

    model = cargar_modelo(args.horizon)
    detalle = []

    for bloque in range(args.blocks):
        train_end = inicio_test + bloque * args.horizon
        test_end = train_end + args.horizon

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy().reset_index(drop=True)

        contexto = train.tail(MAX_CONTEXT)
        series = [contexto[c].to_numpy(dtype=np.float32) for c in columnas]
        baseline_freq = top_k_frecuencia(
            train, columnas, ventana=args.freq_window, k=TOP_K
        )

        print(
            f"Bloque {bloque + 1}/{args.blocks}: "
            f"{test['Fecha'].min().date()} a {test['Fecha'].max().date()} "
            f"(contexto={len(contexto)})"
        )

        point, _ = model.forecast(horizon=args.horizon, inputs=series)

        for paso in range(args.horizon):
            pred_timesfm = top_k_scores(point[:, paso], columnas)
            reales = numeros_reales(test.loc[paso], columnas)

            hit_timesfm = aciertos(pred_timesfm, reales)
            hit_freq = aciertos(baseline_freq, reales)

            detalle.append(
                {
                    "Bloque": bloque + 1,
                    "PasoEnBloque": paso + 1,
                    "FechaReal": test.loc[paso, "Fecha"].date().isoformat(),
                    "TimesFMTop6": formato_numeros(pred_timesfm),
                    "FrecuenciaTop6": formato_numeros(baseline_freq),
                    "NumerosReales": formato_numeros(reales),
                    "AciertosTimesFM": len(hit_timesfm),
                    "NumerosAcertadosTimesFM": formato_numeros(hit_timesfm),
                    "AciertosFrecuencia": len(hit_freq),
                    "NumerosAcertadosFrecuencia": formato_numeros(hit_freq),
                }
            )

    detalle_df = pd.DataFrame(detalle)
    detalle_df.to_csv(DETAIL_FILE, index=False)

    n = len(detalle_df)
    prom_timesfm = float(detalle_df["AciertosTimesFM"].mean())
    prom_freq = float(detalle_df["AciertosFrecuencia"].mean())
    media_azar, var_azar, se_azar = estadistica_azar(n)
    z = (prom_timesfm - media_azar) / se_azar if se_azar > 0 else float("nan")
    p_aprox = p_value_normal(z) if math.isfinite(z) else float("nan")

    resumen = pd.DataFrame(
        [
            {
                "SorteosEvaluados": n,
                "Bloques": args.blocks,
                "HorizontePorBloque": args.horizon,
                "VentanaFrecuencia": args.freq_window,
                "AciertosTimesFMTotales": int(detalle_df["AciertosTimesFM"].sum()),
                "AciertosTimesFMPromedio": prom_timesfm,
                "AciertosFrecuenciaTotales": int(
                    detalle_df["AciertosFrecuencia"].sum()
                ),
                "AciertosFrecuenciaPromedio": prom_freq,
                "EsperadoAzarPromedioTop6": media_azar,
                "EsperadoAzarTotal": media_azar * n,
                "VarianzaAzarPorSorteo": var_azar,
                "ErrorEstandarMediaAzar": se_azar,
                "DeltaTimesFMVsAzar": prom_timesfm - media_azar,
                "DeltaTimesFMVsFrecuencia": prom_timesfm - prom_freq,
                "ZTimesFMVsAzarAprox": z,
                "PValueBilateralNormalAprox": p_aprox,
                "TimesFMSorteos0Aciertos": int(
                    (detalle_df["AciertosTimesFM"] == 0).sum()
                ),
                "TimesFMSorteos1OMas": int(
                    (detalle_df["AciertosTimesFM"] >= 1).sum()
                ),
                "TimesFMMaximoAciertos": int(
                    detalle_df["AciertosTimesFM"].max()
                ),
            }
        ]
    )
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("-" * 68)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"TimesFM promedio Top-6: {prom_timesfm:.4f}")
    print(f"Frecuencia reciente promedio Top-6: {prom_freq:.4f}")
    print(f"Referencia aleatoria teorica Top-6: {media_azar:.4f}")
    print(f"Delta TimesFM vs azar: {prom_timesfm - media_azar:+.4f}")
    print(f"Delta TimesFM vs frecuencia: {prom_timesfm - prom_freq:+.4f}")
    print(f"z aproximado TimesFM vs azar: {z:.3f}")
    print(f"p bilateral aproximado: {p_aprox:.4f}")
    print(
        "Nota: el p-value es una aproximacion diagnostica. Los bloques de "
        "forecast multi-paso no deben interpretarse como ensayos totalmente independientes."
    )


if __name__ == "__main__":
    main()
