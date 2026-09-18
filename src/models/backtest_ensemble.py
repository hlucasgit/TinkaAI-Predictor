"""Backtest walk-forward de TinkaAI Ensemble.

Compara, sobre los ultimos N sorteos:
- TimesFM Top-6
- Frecuencia reciente Top-6
- Ensemble Top-6

El Ensemble combina scores normalizados de TimesFM, frecuencia historica,
frecuencia reciente y recencia. Los pesos se pasan por CLI y deben sumar 1.
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
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_ensemble_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_ensemble_resumen.csv"

TOTAL_NUMEROS = 53
TOP_K = 6
DEFAULT_N_SORTEOS = 128
DEFAULT_CONTEXT = 2048
DEFAULT_RECENT_WINDOW = 100

DEFAULT_W_TIMESFM = 0.50
DEFAULT_W_HIST = 0.20
DEFAULT_W_RECENT = 0.30
DEFAULT_W_RECENCY = 0.00


def minmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    minimo = float(np.nanmin(x))
    maximo = float(np.nanmax(x))
    if not np.isfinite(minimo) or not np.isfinite(maximo):
        raise ValueError("Valores no finitos al normalizar")
    if abs(maximo - minimo) < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - minimo) / (maximo - minimo)


def validar_pesos(w_timesfm: float, w_hist: float, w_recent: float, w_recency: float) -> None:
    pesos = np.array([w_timesfm, w_hist, w_recent, w_recency], dtype=float)
    if (pesos < 0).any():
        raise ValueError("Los pesos no pueden ser negativos")
    if not np.isclose(pesos.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Los pesos deben sumar 1.0; suma={pesos.sum():.6f}")


def cargar_datos() -> tuple[pd.DataFrame, list[str]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if "Fecha" not in df.columns:
        raise ValueError("timesfm_input.csv requiere Fecha")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    if not np.isin(df[columnas].to_numpy(dtype=int), [0, 1]).all():
        raise ValueError("El dataset debe ser binario 0/1")

    return df, columnas


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


def recencia_scores(contexto: pd.DataFrame, columnas: list[str]) -> np.ndarray:
    n = len(contexto)
    gaps = []
    for c in columnas:
        idx = np.flatnonzero(contexto[c].to_numpy(dtype=int) == 1)
        gaps.append(float(n if len(idx) == 0 else (n - 1) - int(idx[-1])))
    return np.asarray(gaps, dtype=float)


def ensemble_scores(
    contexto: pd.DataFrame,
    columnas: list[str],
    scores_tfm_raw: np.ndarray,
    recent_window: int,
    w_timesfm: float,
    w_hist: float,
    w_recent: float,
    w_recency: float,
) -> np.ndarray:
    hist_raw = contexto[columnas].mean(axis=0).to_numpy(dtype=float)
    reciente = contexto.iloc[-min(recent_window, len(contexto)):]
    recent_raw = reciente[columnas].mean(axis=0).to_numpy(dtype=float)
    recency_raw = recencia_scores(contexto, columnas)

    return (
        w_timesfm * minmax(scores_tfm_raw)
        + w_hist * minmax(hist_raw)
        + w_recent * minmax(recent_raw)
        + w_recency * minmax(recency_raw)
    )


def top_k(scores: np.ndarray, columnas: list[str]) -> list[int]:
    idx = np.argsort(np.asarray(scores, dtype=float))[::-1][:TOP_K]
    return sorted(int(columnas[i][1:]) for i in idx)


def numeros_reales(fila: pd.Series, columnas: list[str]) -> list[int]:
    return sorted(int(c[1:]) for c in columnas if int(fila[c]) == 1)


def hits(pred: list[int], reales: list[int]) -> tuple[int, list[int]]:
    aciertos = sorted(set(pred) & set(reales))
    return len(aciertos), aciertos


def fmt(nums: list[int]) -> str:
    return "-".join(f"{n:02d}" for n in nums)


def pmf_hits_azar() -> np.ndarray:
    den = math.comb(TOTAL_NUMEROS, TOP_K)
    probs = []
    for x in range(TOP_K + 1):
        num = math.comb(TOP_K, x) * math.comb(TOTAL_NUMEROS - TOP_K, TOP_K - x)
        probs.append(num / den)
    return np.asarray(probs, dtype=float)


def p_valor_azar_total(aciertos_totales: int, n_sorteos: int) -> float:
    dist = np.array([1.0], dtype=float)
    pmf = pmf_hits_azar()
    for _ in range(n_sorteos):
        dist = np.convolve(dist, pmf)
    return float(dist[aciertos_totales:].sum())


def p_valor_pareado_signflip(diferencias: np.ndarray, repeticiones: int = 30000) -> float:
    diferencias = np.asarray(diferencias, dtype=float)
    observado = abs(float(diferencias.mean()))
    if observado == 0:
        return 1.0

    rng = np.random.default_rng(20260917)
    extremos = 0
    lote = 2500
    hechos = 0

    while hechos < repeticiones:
        n = min(lote, repeticiones - hechos)
        signos = rng.choice(np.array([-1.0, 1.0]), size=(n, len(diferencias)))
        medias = np.abs((signos * diferencias).mean(axis=1))
        extremos += int((medias >= observado - 1e-12).sum())
        hechos += n

    return (extremos + 1.0) / (repeticiones + 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Backtest walk-forward TinkaAI Ensemble")
    p.add_argument("--n-sorteos", type=int, default=DEFAULT_N_SORTEOS)
    p.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW)
    p.add_argument("--w-timesfm", type=float, default=DEFAULT_W_TIMESFM)
    p.add_argument("--w-hist", type=float, default=DEFAULT_W_HIST)
    p.add_argument("--w-recent", type=float, default=DEFAULT_W_RECENT)
    p.add_argument("--w-recency", type=float, default=DEFAULT_W_RECENCY)
    return p.parse_args()


def main():
    args = parse_args()
    validar_pesos(args.w_timesfm, args.w_hist, args.w_recent, args.w_recency)

    if args.n_sorteos < 1:
        raise ValueError("--n-sorteos debe ser >= 1")
    if args.context < 32:
        raise ValueError("--context debe ser >= 32")
    if args.recent_window < 1:
        raise ValueError("--recent-window debe ser >= 1")

    print("=" * 76)
    print("TinkaAI Predictor - Backtest Ensemble")
    print("=" * 76)

    df, columnas = cargar_datos()
    if len(df) <= args.n_sorteos:
        raise ValueError("No hay suficientes sorteos")

    inicio = len(df) - args.n_sorteos
    periodo = df.iloc[inicio:]

    print(f"Sorteos historicos: {len(df)}")
    print(f"Sorteos evaluados: {args.n_sorteos}")
    print(f"Periodo: {periodo['Fecha'].min().date()} a {periodo['Fecha'].max().date()}")
    print(f"Contexto TimesFM: {args.context}")
    print(f"Frecuencia reciente: {args.recent_window}")
    print(
        "Pesos Ensemble: "
        f"TimesFM={args.w_timesfm:.2f}, "
        f"Hist={args.w_hist:.2f}, "
        f"Reciente={args.w_recent:.2f}, "
        f"Recencia={args.w_recency:.2f}"
    )
    print("Cargando modelo TimesFM...")

    model = cargar_modelo(args.context)
    detalle = []

    for j, idx in enumerate(range(inicio, len(df)), start=1):
        contexto = df.iloc[:idx]
        contexto_tfm = contexto.iloc[-min(args.context, len(contexto)):]
        real = df.iloc[idx]

        series = [contexto_tfm[c].to_numpy(dtype=np.float32) for c in columnas]
        point, _ = model.forecast(horizon=1, inputs=series)
        scores_tfm_raw = np.asarray(point)[:, 0]

        scores_ens = ensemble_scores(
            contexto=contexto,
            columnas=columnas,
            scores_tfm_raw=scores_tfm_raw,
            recent_window=args.recent_window,
            w_timesfm=args.w_timesfm,
            w_hist=args.w_hist,
            w_recent=args.w_recent,
            w_recency=args.w_recency,
        )

        recent_raw = (
            contexto.iloc[-min(args.recent_window, len(contexto)):][columnas]
            .mean(axis=0)
            .to_numpy(dtype=float)
        )

        pred_tfm = top_k(scores_tfm_raw, columnas)
        pred_freq = top_k(recent_raw, columnas)
        pred_ens = top_k(scores_ens, columnas)
        reales = numeros_reales(real, columnas)

        hit_tfm, ac_tfm = hits(pred_tfm, reales)
        hit_freq, ac_freq = hits(pred_freq, reales)
        hit_ens, ac_ens = hits(pred_ens, reales)

        detalle.append(
            {
                "NroEvaluacion": j,
                "FechaReal": real["Fecha"].date().isoformat(),
                "PrediccionTimesFMTop6": fmt(pred_tfm),
                "AciertosTimesFM": hit_tfm,
                "PrediccionFrecuenciaTop6": fmt(pred_freq),
                "AciertosFrecuencia": hit_freq,
                "PrediccionEnsembleTop6": fmt(pred_ens),
                "AciertosEnsemble": hit_ens,
                "NumerosAcertadosEnsemble": fmt(ac_ens),
                "NumerosReales": fmt(reales),
            }
        )

        if j == 1 or j % 8 == 0 or j == args.n_sorteos:
            print(
                f"[{j:>3}/{args.n_sorteos}] {real['Fecha'].date()} | "
                f"TimesFM={hit_tfm} | Frec={hit_freq} | Ensemble={hit_ens}"
            )

    d = pd.DataFrame(detalle)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(DETAIL_FILE, index=False)

    h_tfm = d["AciertosTimesFM"].to_numpy(dtype=int)
    h_freq = d["AciertosFrecuencia"].to_numpy(dtype=int)
    h_ens = d["AciertosEnsemble"].to_numpy(dtype=int)

    n = len(d)
    total_tfm = int(h_tfm.sum())
    total_freq = int(h_freq.sum())
    total_ens = int(h_ens.sum())

    prom_tfm = float(h_tfm.mean())
    prom_freq = float(h_freq.mean())
    prom_ens = float(h_ens.mean())
    esperado = TOP_K * TOP_K / TOTAL_NUMEROS

    resumen = pd.DataFrame(
        [
            {
                "SorteosEvaluados": n,
                "PeriodoInicio": periodo["Fecha"].min().date().isoformat(),
                "PeriodoFin": periodo["Fecha"].max().date().isoformat(),
                "AciertosTotalesTimesFM": total_tfm,
                "PromedioTimesFM": prom_tfm,
                "AciertosTotalesFrecuencia": total_freq,
                "PromedioFrecuencia": prom_freq,
                "AciertosTotalesEnsemble": total_ens,
                "PromedioEnsemble": prom_ens,
                "EsperadoAzarTop6": esperado,
                "DeltaEnsembleVsAzar": prom_ens - esperado,
                "DeltaEnsembleVsTimesFM": prom_ens - prom_tfm,
                "DeltaEnsembleVsFrecuencia": prom_ens - prom_freq,
                "PUnilateralEnsembleVsAzar": p_valor_azar_total(total_ens, n),
                "PPareadoEnsembleVsTimesFM": p_valor_pareado_signflip(h_ens - h_tfm),
                "PPareadoEnsembleVsFrecuencia": p_valor_pareado_signflip(h_ens - h_freq),
                "PesoTimesFM": args.w_timesfm,
                "PesoHist": args.w_hist,
                "PesoReciente": args.w_recent,
                "PesoRecencia": args.w_recency,
            }
        ]
    )
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("-" * 76)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"TimesFM promedio Top-6: {prom_tfm:.4f} ({total_tfm})")
    print(f"Frecuencia promedio Top-6: {prom_freq:.4f} ({total_freq})")
    print(f"Ensemble promedio Top-6: {prom_ens:.4f} ({total_ens})")
    print(f"Referencia aleatoria: {esperado:.4f}")
    print(f"Delta Ensemble vs azar: {prom_ens - esperado:+.4f}")
    print(f"Delta Ensemble vs TimesFM: {prom_ens - prom_tfm:+.4f}")
    print(f"Delta Ensemble vs frecuencia: {prom_ens - prom_freq:+.4f}")
    print(
        "p unilateral Ensemble vs azar: "
        f"{float(resumen.loc[0, 'PUnilateralEnsembleVsAzar']):.6f}"
    )
    print(
        "p pareado Ensemble vs TimesFM: "
        f"{float(resumen.loc[0, 'PPareadoEnsembleVsTimesFM']):.6f}"
    )
    print(
        "p pareado Ensemble vs frecuencia: "
        f"{float(resumen.loc[0, 'PPareadoEnsembleVsFrecuencia']):.6f}"
    )


if __name__ == "__main__":
    main()
