"""Backtest walk-forward de un paso para TimesFM.

Evalua los ultimos N sorteos usando un esquema realista de produccion:
para cada sorteo, el modelo solo ve sorteos anteriores, pronostica UN paso,
y luego avanza al siguiente corte temporal.

Compara TimesFM contra una linea base simple de frecuencia reciente y contra
la referencia teorica de elegir 6 numeros al azar entre 53.
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
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_walkforward_timesfm_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_walkforward_timesfm_resumen.csv"

TOP_K = 6
TOTAL_NUMEROS = 53
DEFAULT_N_SORTEOS = 64
DEFAULT_CONTEXT = 2048
DEFAULT_FREQ_WINDOW = 100


def cargar_datos() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if "Fecha" not in df.columns:
        raise ValueError("timesfm_input.csv requiere la columna Fecha")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha").reset_index(drop=True)


def cargar_modelo(max_context: int):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=1,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def top_k_desde_scores(scores: np.ndarray, columnas: list[str], k: int = TOP_K) -> list[int]:
    scores = np.asarray(scores, dtype=float)
    indices = np.argsort(scores)[::-1][:k]
    return sorted(int(columnas[i][1:]) for i in indices)


def top_k_frecuencia(contexto: pd.DataFrame, columnas: list[str], ventana: int) -> list[int]:
    reciente = contexto.iloc[-min(ventana, len(contexto)):]
    frecuencias = reciente[columnas].sum(axis=0).to_numpy(dtype=float)
    return top_k_desde_scores(frecuencias, columnas)


def numeros_reales(fila: pd.Series, columnas: list[str]) -> list[int]:
    return sorted(int(c[1:]) for c in columnas if int(fila[c]) == 1)


def hits(predichos: list[int], reales: list[int]) -> tuple[int, list[int]]:
    aciertos = sorted(set(predichos) & set(reales))
    return len(aciertos), aciertos


def pmf_hits_azar() -> np.ndarray:
    """PMF hipergeometrica de aciertos al elegir 6 numeros de 53."""
    den = math.comb(TOTAL_NUMEROS, TOP_K)
    probs = []
    for x in range(TOP_K + 1):
        if TOP_K - x > TOTAL_NUMEROS - TOP_K:
            probs.append(0.0)
            continue
        num = math.comb(TOP_K, x) * math.comb(TOTAL_NUMEROS - TOP_K, TOP_K - x)
        probs.append(num / den)
    return np.asarray(probs, dtype=float)


def p_valor_azar_total(aciertos_totales: int, n_sorteos: int) -> float:
    """P(X >= observado) bajo referencia aleatoria, asumiendo sorteos independientes."""
    dist = np.array([1.0], dtype=float)
    pmf = pmf_hits_azar()
    for _ in range(n_sorteos):
        dist = np.convolve(dist, pmf)
    return float(dist[aciertos_totales:].sum())


def p_valor_pareado_signflip(diferencias: np.ndarray, repeticiones: int = 20000) -> float:
    """Prueba de permutacion por cambio de signo para diferencia media pareada."""
    diferencias = np.asarray(diferencias, dtype=float)
    if len(diferencias) == 0:
        return float("nan")

    observado = abs(float(diferencias.mean()))
    if observado == 0:
        return 1.0

    rng = np.random.default_rng(20260916)
    extremos = 0
    lote = 2000
    hechos = 0

    while hechos < repeticiones:
        n = min(lote, repeticiones - hechos)
        signos = rng.choice(np.array([-1.0, 1.0]), size=(n, len(diferencias)))
        medias = np.abs((signos * diferencias).mean(axis=1))
        extremos += int((medias >= observado - 1e-12).sum())
        hechos += n

    return (extremos + 1.0) / (repeticiones + 1.0)


def formatear_numeros(numeros: list[int]) -> str:
    return "-".join(f"{n:02d}" for n in numeros)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest walk-forward TimesFM para TinkaAI")
    parser.add_argument(
        "--n-sorteos",
        type=int,
        default=DEFAULT_N_SORTEOS,
        help=f"Cantidad de sorteos finales a evaluar (default: {DEFAULT_N_SORTEOS})",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help=f"Maximo de sorteos de contexto para TimesFM (default: {DEFAULT_CONTEXT})",
    )
    parser.add_argument(
        "--freq-window",
        type=int,
        default=DEFAULT_FREQ_WINDOW,
        help=f"Ventana de sorteos para baseline de frecuencia (default: {DEFAULT_FREQ_WINDOW})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_sorteos < 1:
        raise ValueError("--n-sorteos debe ser >= 1")
    if args.context < 32:
        raise ValueError("--context debe ser >= 32")
    if args.freq_window < 1:
        raise ValueError("--freq-window debe ser >= 1")

    print("=" * 72)
    print("TinkaAI Predictor - Backtest Walk-Forward TimesFM")
    print("=" * 72)

    df = cargar_datos()
    columnas = [c for c in df.columns if c.startswith("N")]

    if len(columnas) != TOTAL_NUMEROS:
        raise ValueError(
            f"Se esperaban {TOTAL_NUMEROS} series N01..N53 y se encontraron {len(columnas)}"
        )
    if len(df) <= args.n_sorteos:
        raise ValueError("No hay suficientes sorteos para el backtest solicitado")

    inicio_test = len(df) - args.n_sorteos
    periodo = df.iloc[inicio_test:]

    print(f"Sorteos historicos: {len(df)}")
    print(f"Sorteos walk-forward a evaluar: {args.n_sorteos}")
    print(f"Contexto maximo TimesFM: {args.context}")
    print(f"Baseline frecuencia: ultimos {args.freq_window} sorteos")
    print(
        f"Periodo evaluado: {periodo['Fecha'].min().date()} a {periodo['Fecha'].max().date()}"
    )
    print("Cargando modelo TimesFM una sola vez...")

    model = cargar_modelo(args.context)

    detalle = []

    for j, idx in enumerate(range(inicio_test, len(df)), start=1):
        contexto_completo = df.iloc[:idx]
        contexto_tfm = contexto_completo.iloc[-min(args.context, len(contexto_completo)):]
        real = df.iloc[idx]

        series = [contexto_tfm[c].to_numpy(dtype=np.float32) for c in columnas]
        point, _ = model.forecast(horizon=1, inputs=series)
        scores_tfm = np.asarray(point)[:, 0]

        pred_tfm = top_k_desde_scores(scores_tfm, columnas)
        pred_freq = top_k_frecuencia(contexto_completo, columnas, args.freq_window)
        reales = numeros_reales(real, columnas)

        hit_tfm, acertados_tfm = hits(pred_tfm, reales)
        hit_freq, acertados_freq = hits(pred_freq, reales)

        detalle.append(
            {
                "NroEvaluacion": j,
                "FechaReal": real["Fecha"].date().isoformat(),
                "ContextoUsadoTimesFM": len(contexto_tfm),
                "PrediccionTimesFMTop6": formatear_numeros(pred_tfm),
                "AciertosTimesFM": hit_tfm,
                "NumerosAcertadosTimesFM": formatear_numeros(acertados_tfm),
                f"PrediccionFrecuencia{args.freq_window}Top6": formatear_numeros(pred_freq),
                f"AciertosFrecuencia{args.freq_window}": hit_freq,
                f"NumerosAcertadosFrecuencia{args.freq_window}": formatear_numeros(acertados_freq),
                "NumerosReales": formatear_numeros(reales),
            }
        )

        if j == 1 or j % 8 == 0 or j == args.n_sorteos:
            print(
                f"[{j:>3}/{args.n_sorteos}] {real['Fecha'].date()} | "
                f"TimesFM={hit_tfm} | Frecuencia={hit_freq}"
            )

    detalle_df = pd.DataFrame(detalle)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detalle_df.to_csv(DETAIL_FILE, index=False)

    col_freq_hits = f"AciertosFrecuencia{args.freq_window}"
    hits_tfm = detalle_df["AciertosTimesFM"].to_numpy(dtype=int)
    hits_freq = detalle_df[col_freq_hits].to_numpy(dtype=int)

    total_tfm = int(hits_tfm.sum())
    total_freq = int(hits_freq.sum())
    promedio_tfm = float(hits_tfm.mean())
    promedio_freq = float(hits_freq.mean())
    esperado_azar = TOP_K * TOP_K / TOTAL_NUMEROS

    p_tfm_azar = p_valor_azar_total(total_tfm, args.n_sorteos)
    p_freq_azar = p_valor_azar_total(total_freq, args.n_sorteos)
    p_pareado = p_valor_pareado_signflip(hits_tfm - hits_freq)

    resumen = {
        "SorteosEvaluados": args.n_sorteos,
        "PeriodoInicio": periodo["Fecha"].min().date().isoformat(),
        "PeriodoFin": periodo["Fecha"].max().date().isoformat(),
        "ContextoMaximoTimesFM": args.context,
        "VentanaFrecuencia": args.freq_window,
        "AciertosTotalesTimesFM": total_tfm,
        "AciertosPromedioTimesFM": promedio_tfm,
        "AciertosTotalesFrecuencia": total_freq,
        "AciertosPromedioFrecuencia": promedio_freq,
        "EsperadoAzarTop6": esperado_azar,
        "DeltaTimesFMVsAzar": promedio_tfm - esperado_azar,
        "DeltaFrecuenciaVsAzar": promedio_freq - esperado_azar,
        "DeltaTimesFMVsFrecuencia": promedio_tfm - promedio_freq,
        "PUnilateralTimesFMVsAzar": p_tfm_azar,
        "PUnilateralFrecuenciaVsAzar": p_freq_azar,
        "PPareadoTimesFMVsFrecuencia": p_pareado,
        "SorteosTimesFM0Aciertos": int((hits_tfm == 0).sum()),
        "SorteosTimesFM1Acierto": int((hits_tfm == 1).sum()),
        "SorteosTimesFM2OMas": int((hits_tfm >= 2).sum()),
        "MaxAciertosTimesFM": int(hits_tfm.max()),
        "SorteosFrecuencia0Aciertos": int((hits_freq == 0).sum()),
        "SorteosFrecuencia1Acierto": int((hits_freq == 1).sum()),
        "SorteosFrecuencia2OMas": int((hits_freq >= 2).sum()),
        "MaxAciertosFrecuencia": int(hits_freq.max()),
    }

    pd.DataFrame([resumen]).to_csv(SUMMARY_FILE, index=False)

    print("-" * 72)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"TimesFM promedio Top-6: {promedio_tfm:.4f} ({total_tfm} aciertos totales)")
    print(
        f"Frecuencia {args.freq_window} promedio Top-6: "
        f"{promedio_freq:.4f} ({total_freq} aciertos totales)"
    )
    print(f"Referencia aleatoria teorica Top-6: {esperado_azar:.4f}")
    print(f"Delta TimesFM vs azar: {promedio_tfm - esperado_azar:+.4f}")
    print(f"Delta TimesFM vs frecuencia: {promedio_tfm - promedio_freq:+.4f}")
    print(f"p unilateral exacto TimesFM vs azar*: {p_tfm_azar:.6f}")
    print(f"p unilateral exacto frecuencia vs azar*: {p_freq_azar:.6f}")
    print(f"p pareado TimesFM vs frecuencia**: {p_pareado:.6f}")
    print("* Bajo el supuesto de sorteos independientes y 6 numeros equiprobables entre 53.")
    print("** Permutacion por cambio de signo; diagnostico, no prueba de causalidad.")


if __name__ == "__main__":
    main()
