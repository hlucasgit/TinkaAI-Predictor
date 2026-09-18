"""Holdout out-of-time del Ensemble TinkaAI sobre sorteos posteriores al 10/06/2026.

Los pesos se leen por defecto de:
data/processed/optimizacion_pesos_ensemble_resumen.csv

Asi, la seleccion de pesos queda congelada ANTES de evaluar el holdout.
El holdout nunca se usa para optimizar pesos.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm


ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
HOLDOUT_FILE = ROOT / "data" / "raw" / "tinka_actualizacion_2026.csv"
WEIGHTS_FILE = ROOT / "data" / "processed" / "optimizacion_pesos_ensemble_resumen.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_holdout_ensemble_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_holdout_ensemble_resumen.csv"

CUTOFF = pd.Timestamp("2026-06-10")
TOTAL_NUMEROS = 53
TOP_K = 6
MAX_CONTEXT = 2048
FREQ_WINDOW = 100
TIE_TOL = 1e-12


def normalizar_rango(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Valores no finitos al normalizar")
    if len(x) <= 1:
        return np.full_like(x, 0.5, dtype=float)
    ranks = pd.Series(x).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 1.0) / (len(x) - 1.0)


def cargar_pesos(args) -> tuple[float, float, float]:
    if args.w_timesfm is not None or args.w_hist is not None or args.w_recent is not None:
        if args.w_timesfm is None or args.w_hist is None or args.w_recent is None:
            raise ValueError("Si pasa pesos manuales, debe indicar los tres")
        pesos = (args.w_timesfm, args.w_hist, args.w_recent)
    else:
        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"No existe {WEIGHTS_FILE}. Ejecute primero optimize_ensemble_weights.py"
            )
        r = pd.read_csv(WEIGHTS_FILE)
        if r.empty:
            raise ValueError("El resumen de optimizacion esta vacio")
        fila = r.iloc[0]
        pesos = (
            float(fila["PesoTimesFM"]),
            float(fila["PesoHistorico"]),
            float(fila["PesoReciente"]),
        )

    if any(p < 0 for p in pesos) or not np.isclose(sum(pesos), 1.0, atol=1e-9):
        raise ValueError(f"Pesos invalidos: {pesos}; deben ser >=0 y sumar 1")
    return pesos


def cargar_base() -> pd.DataFrame:
    if not BASE_FILE.exists():
        raise FileNotFoundError(BASE_FILE)
    df = pd.read_csv(BASE_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df[df["Fecha"] <= CUTOFF].sort_values("Fecha").reset_index(drop=True)


def cargar_holdout() -> pd.DataFrame:
    if not HOLDOUT_FILE.exists():
        raise FileNotFoundError(HOLDOUT_FILE)

    raw = pd.read_csv(HOLDOUT_FILE)
    req = ["IdSorteo", "Fecha"] + [f"Numero{i}" for i in range(1, 7)]
    faltan = [c for c in req if c not in raw.columns]
    if faltan:
        raise ValueError(f"Faltan columnas holdout: {faltan}")

    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    raw = raw.sort_values("Fecha").reset_index(drop=True)
    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    filas = []

    for _, r in raw.iterrows():
        nums = [int(r[f"Numero{i}"]) for i in range(1, 7)]
        if len(set(nums)) != TOP_K or min(nums) < 1 or max(nums) > TOTAL_NUMEROS:
            raise ValueError(f"Sorteo invalido: {r['IdSorteo']}")

        fila = {"Fecha": r["Fecha"], "IdSorteoOficial": int(r["IdSorteo"])}
        for c in columnas:
            fila[c] = 0
        for n in nums:
            fila[f"N{n:02d}"] = 1
        filas.append(fila)

    out = pd.DataFrame(filas)
    if (out["Fecha"] <= CUTOFF).any():
        raise ValueError("El holdout contiene fechas <= cutoff")
    return out


def cargar_modelo():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=1,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def score_componentes(
    historial: pd.DataFrame,
    columnas: list[str],
    tfm_raw: np.ndarray,
    w_tfm: float,
    w_hist: float,
    w_recent: float,
) -> tuple[np.ndarray, np.ndarray]:
    hist_raw = historial[columnas].mean(axis=0).to_numpy(dtype=float)
    recent_raw = (
        historial.iloc[-min(FREQ_WINDOW, len(historial)):][columnas]
        .mean(axis=0)
        .to_numpy(dtype=float)
    )

    ensemble = (
        w_tfm * normalizar_rango(tfm_raw)
        + w_hist * normalizar_rango(hist_raw)
        + w_recent * normalizar_rango(recent_raw)
    )
    return ensemble, recent_raw


def top6_determinista(scores: np.ndarray, numeros: np.ndarray) -> list[int]:
    order = np.lexsort((numeros, -scores))
    return sorted(numeros[order[:TOP_K]].astype(int).tolist())


def hits_esperados_empate(scores: np.ndarray, real_bin: np.ndarray) -> float:
    threshold = float(np.sort(scores)[-TOP_K])
    superiores = scores > threshold + TIE_TOL
    empatados = np.isclose(scores, threshold, rtol=0.0, atol=TIE_TOL)
    espacios = TOP_K - int(superiores.sum())

    hits_sup = float(real_bin[superiores].sum())
    if espacios == 0:
        return hits_sup

    n_emp = int(empatados.sum())
    if n_emp == 0:
        raise RuntimeError("Empate de corte inconsistente")
    return hits_sup + espacios * float(real_bin[empatados].sum()) / n_emp


def fmt(nums: list[int]) -> str:
    return "-".join(f"{n:02d}" for n in nums)


def pmf_hits_azar() -> np.ndarray:
    den = math.comb(TOTAL_NUMEROS, TOP_K)
    return np.asarray(
        [
            math.comb(TOP_K, x)
            * math.comb(TOTAL_NUMEROS - TOP_K, TOP_K - x)
            / den
            for x in range(TOP_K + 1)
        ],
        dtype=float,
    )


def p_valor_azar_total(aciertos_totales: int, n_sorteos: int) -> float:
    dist = np.array([1.0], dtype=float)
    pmf = pmf_hits_azar()
    for _ in range(n_sorteos):
        dist = np.convolve(dist, pmf)
    return float(dist[aciertos_totales:].sum())


def parse_args():
    p = argparse.ArgumentParser(description="Holdout externo del Ensemble TinkaAI")
    p.add_argument("--w-timesfm", type=float, default=None)
    p.add_argument("--w-hist", type=float, default=None)
    p.add_argument("--w-recent", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    w_tfm, w_hist, w_recent = cargar_pesos(args)

    print("=" * 78)
    print("TinkaAI Predictor - Holdout externo Ensemble 2026")
    print("=" * 78)
    print(
        f"Pesos congelados: TimesFM={w_tfm:.2f}, "
        f"Historico={w_hist:.2f}, Reciente={w_recent:.2f}"
    )

    base = cargar_base()
    holdout = cargar_holdout()
    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    numeros = np.arange(1, TOTAL_NUMEROS + 1, dtype=int)

    print(f"Base hasta: {CUTOFF.date()} | sorteos={len(base)}")
    print(
        f"Holdout: {len(holdout)} sorteos | "
        f"{holdout['Fecha'].min().date()} a {holdout['Fecha'].max().date()}"
    )
    print("Cargando TimesFM...")

    model = cargar_modelo()
    historial = base[["Fecha"] + columnas].copy()
    filas = []

    for i, (_, real) in enumerate(holdout.iterrows(), start=1):
        contexto_tfm = historial.iloc[-min(MAX_CONTEXT, len(historial)):]
        series = [contexto_tfm[c].to_numpy(dtype=np.float32) for c in columnas]
        point, _ = model.forecast(horizon=1, inputs=series)
        tfm_raw = np.asarray(point)[:, 0]

        ens, freq_raw = score_componentes(
            historial, columnas, tfm_raw, w_tfm, w_hist, w_recent
        )

        real_bin = real[columnas].to_numpy(dtype=int)
        reales = sorted(numeros[real_bin == 1].tolist())

        pred_tfm = top6_determinista(tfm_raw, numeros)
        pred_freq = top6_determinista(freq_raw, numeros)
        pred_ens = top6_determinista(ens, numeros)

        h_tfm = len(set(pred_tfm) & set(reales))
        h_freq = len(set(pred_freq) & set(reales))
        h_ens = len(set(pred_ens) & set(reales))

        filas.append(
            {
                "NroEvaluacion": i,
                "IdSorteoOficial": int(real["IdSorteoOficial"]),
                "FechaReal": real["Fecha"].date().isoformat(),
                "PredTimesFM": fmt(pred_tfm),
                "AciertosTimesFM": h_tfm,
                "EsperadoTimesFMConEmpates": hits_esperados_empate(tfm_raw, real_bin),
                "PredFrecuencia100": fmt(pred_freq),
                "AciertosFrecuencia100": h_freq,
                "EsperadoFrecuenciaConEmpates": hits_esperados_empate(freq_raw, real_bin),
                "PredEnsemble": fmt(pred_ens),
                "AciertosEnsemble": h_ens,
                "EsperadoEnsembleConEmpates": hits_esperados_empate(ens, real_bin),
                "NumerosReales": fmt(reales),
            }
        )

        nueva = {"Fecha": real["Fecha"]}
        for c in columnas:
            nueva[c] = int(real[c])
        historial = pd.concat([historial, pd.DataFrame([nueva])], ignore_index=True)

        print(
            f"[{i:>2}/{len(holdout)}] {real['Fecha'].date()} | "
            f"TFM={h_tfm} | Freq={h_freq} | Ens={h_ens}"
        )

    d = pd.DataFrame(filas)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(DETAIL_FILE, index=False)

    n = len(d)
    esperado_azar = TOP_K * TOP_K / TOTAL_NUMEROS

    total_tfm = int(d["AciertosTimesFM"].sum())
    total_freq = int(d["AciertosFrecuencia100"].sum())
    total_ens = int(d["AciertosEnsemble"].sum())

    prom_tfm = float(d["AciertosTimesFM"].mean())
    prom_freq = float(d["AciertosFrecuencia100"].mean())
    prom_ens = float(d["AciertosEnsemble"].mean())

    exp_tfm = float(d["EsperadoTimesFMConEmpates"].mean())
    exp_freq = float(d["EsperadoFrecuenciaConEmpates"].mean())
    exp_ens = float(d["EsperadoEnsembleConEmpates"].mean())

    resumen = pd.DataFrame(
        [
            {
                "TipoEvaluacion": "out-of-time walk-forward",
                "CutoffHistorico": CUTOFF.date().isoformat(),
                "SorteosEvaluados": n,
                "PesoTimesFM": w_tfm,
                "PesoHistorico": w_hist,
                "PesoReciente": w_recent,
                "PromedioTimesFMDeterminista": prom_tfm,
                "PromedioFrecuenciaDeterminista": prom_freq,
                "PromedioEnsembleDeterminista": prom_ens,
                "PromedioTimesFMTieAware": exp_tfm,
                "PromedioFrecuenciaTieAware": exp_freq,
                "PromedioEnsembleTieAware": exp_ens,
                "EsperadoAzarTop6": esperado_azar,
                "PEnsembleDeterministaVsAzar": p_valor_azar_total(total_ens, n),
                "PTimesFMDeterministaVsAzar": p_valor_azar_total(total_tfm, n),
                "PFrecuenciaDeterministaVsAzar": p_valor_azar_total(total_freq, n),
            }
        ]
    )
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("-" * 78)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"TimesFM determinista:   {prom_tfm:.4f} ({total_tfm})")
    print(f"Frecuencia determinista:{prom_freq:.4f} ({total_freq})")
    print(f"Ensemble determinista:  {prom_ens:.4f} ({total_ens})")
    print(f"TimesFM tie-aware:      {exp_tfm:.4f}")
    print(f"Frecuencia tie-aware:   {exp_freq:.4f}")
    print(f"Ensemble tie-aware:     {exp_ens:.4f}")
    print(f"Azar teorico:           {esperado_azar:.4f}")
    print(
        "p Ensemble determinista vs azar: "
        f"{float(resumen.loc[0, 'PEnsembleDeterministaVsAzar']):.6f}"
    )
    print(
        "El holdout no participa en la seleccion de pesos; "
        "esta es la medicion externa del Ensemble congelado."
    )


if __name__ == "__main__":
    main()
