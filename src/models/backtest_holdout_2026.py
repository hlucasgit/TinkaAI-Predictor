"""Evaluacion out-of-time de TimesFM con sorteos posteriores al 10/06/2026.

Este archivo usa como contexto exclusivamente el dataset historico disponible hasta
2026-06-10 y evalua, en modo walk-forward de un paso, los sorteos oficiales
almacenados en data/raw/tinka_actualizacion_2026.csv.

La finalidad es evitar ajustar el modelo con los mismos sorteos usados para medirlo.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
HOLDOUT_FILE = ROOT / "data" / "raw" / "tinka_actualizacion_2026.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_holdout_2026_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_holdout_2026_resumen.csv"

CUTOFF = pd.Timestamp("2026-06-10")
TOP_K = 6
TOTAL_NUMEROS = 53
MAX_CONTEXT = 2048
FREQ_WINDOW = 100


def cargar_base() -> pd.DataFrame:
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"No existe: {BASE_FILE}")

    df = pd.read_csv(BASE_FILE)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df[df["Fecha"] <= CUTOFF].copy()
    return df.sort_values("Fecha").reset_index(drop=True)


def cargar_holdout() -> pd.DataFrame:
    if not HOLDOUT_FILE.exists():
        raise FileNotFoundError(f"No existe: {HOLDOUT_FILE}")

    raw = pd.read_csv(HOLDOUT_FILE)
    requeridas = ["IdSorteo", "Fecha"] + [f"Numero{i}" for i in range(1, 7)]
    faltantes = [c for c in requeridas if c not in raw.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en holdout: {faltantes}")

    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    raw = raw.sort_values("Fecha").reset_index(drop=True)

    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    filas = []

    for _, r in raw.iterrows():
        numeros = [int(r[f"Numero{i}"]) for i in range(1, 7)]
        if len(set(numeros)) != TOP_K:
            raise ValueError(f"Sorteo {r['IdSorteo']} contiene numeros repetidos")
        if min(numeros) < 1 or max(numeros) > TOTAL_NUMEROS:
            raise ValueError(f"Sorteo {r['IdSorteo']} contiene numeros fuera de 1-53")

        fila = {"Fecha": r["Fecha"], "IdSorteoOficial": int(r["IdSorteo"])}
        for c in columnas:
            fila[c] = 0
        for numero in numeros:
            fila[f"N{numero:02d}"] = 1
        filas.append(fila)

    out = pd.DataFrame(filas)
    if (out["Fecha"] <= CUTOFF).any():
        raise ValueError("El holdout debe contener solo sorteos posteriores al cutoff")
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


def top_k(scores: np.ndarray, columnas: list[str]) -> list[int]:
    idx = np.argsort(np.asarray(scores, dtype=float))[::-1][:TOP_K]
    return sorted(int(columnas[i][1:]) for i in idx)


def top_k_frecuencia(contexto: pd.DataFrame, columnas: list[str]) -> list[int]:
    reciente = contexto.iloc[-min(FREQ_WINDOW, len(contexto)):]
    scores = reciente[columnas].sum(axis=0).to_numpy(dtype=float)
    return top_k(scores, columnas)


def reales_de_fila(fila: pd.Series, columnas: list[str]) -> list[int]:
    return sorted(int(c[1:]) for c in columnas if int(fila[c]) == 1)


def contar_aciertos(pred: list[int], reales: list[int]) -> tuple[int, list[int]]:
    acertados = sorted(set(pred) & set(reales))
    return len(acertados), acertados


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


def p_valor_pareado_signflip(diferencias: np.ndarray, repeticiones: int = 50000) -> float:
    diferencias = np.asarray(diferencias, dtype=float)
    observado = abs(float(diferencias.mean()))
    if observado == 0:
        return 1.0

    rng = np.random.default_rng(20260916)
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


def main():
    print("=" * 76)
    print("TinkaAI Predictor - Holdout Out-of-Time 2026")
    print("=" * 76)

    base = cargar_base()
    holdout = cargar_holdout()
    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]

    faltan_base = [c for c in columnas if c not in base.columns]
    if faltan_base:
        raise ValueError(f"Faltan series en dataset base: {faltan_base}")

    print(f"Contexto congelado hasta: {CUTOFF.date()}")
    print(f"Sorteos base disponibles: {len(base)}")
    print(f"Sorteos holdout: {len(holdout)}")
    print(f"Periodo holdout: {holdout['Fecha'].min().date()} a {holdout['Fecha'].max().date()}")
    print(f"Contexto maximo TimesFM: {MAX_CONTEXT}")
    print(f"Baseline frecuencia: ultimos {FREQ_WINDOW} sorteos")
    print("Cargando modelo TimesFM...")

    model = cargar_modelo()
    historial = base[["Fecha"] + columnas].copy()
    detalle = []

    for i, (_, real) in enumerate(holdout.iterrows(), start=1):
        contexto_tfm = historial.iloc[-min(MAX_CONTEXT, len(historial)):]
        series = [contexto_tfm[c].to_numpy(dtype=np.float32) for c in columnas]
        point, _ = model.forecast(horizon=1, inputs=series)

        pred_tfm = top_k(np.asarray(point)[:, 0], columnas)
        pred_freq = top_k_frecuencia(historial, columnas)
        reales = reales_de_fila(real, columnas)

        hit_tfm, acertados_tfm = contar_aciertos(pred_tfm, reales)
        hit_freq, acertados_freq = contar_aciertos(pred_freq, reales)

        detalle.append({
            "NroEvaluacion": i,
            "IdSorteoOficial": int(real["IdSorteoOficial"]),
            "FechaReal": real["Fecha"].date().isoformat(),
            "PrediccionTimesFMTop6": fmt(pred_tfm),
            "AciertosTimesFM": hit_tfm,
            "NumerosAcertadosTimesFM": fmt(acertados_tfm),
            "PrediccionFrecuencia100Top6": fmt(pred_freq),
            "AciertosFrecuencia100": hit_freq,
            "NumerosAcertadosFrecuencia100": fmt(acertados_freq),
            "NumerosReales": fmt(reales),
        })

        nueva = {"Fecha": real["Fecha"]}
        for c in columnas:
            nueva[c] = int(real[c])
        historial = pd.concat([historial, pd.DataFrame([nueva])], ignore_index=True)

        print(
            f"[{i:>2}/{len(holdout)}] {real['Fecha'].date()} | "
            f"TimesFM={hit_tfm} | Frecuencia={hit_freq}"
        )

    detalle_df = pd.DataFrame(detalle)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detalle_df.to_csv(DETAIL_FILE, index=False)

    hits_tfm = detalle_df["AciertosTimesFM"].to_numpy(dtype=int)
    hits_freq = detalle_df["AciertosFrecuencia100"].to_numpy(dtype=int)
    n = len(detalle_df)
    total_tfm = int(hits_tfm.sum())
    total_freq = int(hits_freq.sum())
    prom_tfm = float(hits_tfm.mean())
    prom_freq = float(hits_freq.mean())
    esperado = TOP_K * TOP_K / TOTAL_NUMEROS

    p_tfm = p_valor_azar_total(total_tfm, n)
    p_freq = p_valor_azar_total(total_freq, n)
    p_pareado = p_valor_pareado_signflip(hits_tfm - hits_freq)

    resumen = pd.DataFrame([{
        "TipoEvaluacion": "out-of-time walk-forward",
        "CutoffHistorico": CUTOFF.date().isoformat(),
        "SorteosEvaluados": n,
        "PeriodoInicio": holdout["Fecha"].min().date().isoformat(),
        "PeriodoFin": holdout["Fecha"].max().date().isoformat(),
        "AciertosTotalesTimesFM": total_tfm,
        "AciertosPromedioTimesFM": prom_tfm,
        "AciertosTotalesFrecuencia100": total_freq,
        "AciertosPromedioFrecuencia100": prom_freq,
        "EsperadoAzarTop6": esperado,
        "DeltaTimesFMVsAzar": prom_tfm - esperado,
        "DeltaTimesFMVsFrecuencia": prom_tfm - prom_freq,
        "PUnilateralTimesFMVsAzar": p_tfm,
        "PUnilateralFrecuenciaVsAzar": p_freq,
        "PPareadoTimesFMVsFrecuencia": p_pareado,
        "MaxAciertosTimesFM": int(hits_tfm.max()),
        "MaxAciertosFrecuencia100": int(hits_freq.max()),
    }])
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("-" * 76)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(f"TimesFM promedio Top-6: {prom_tfm:.4f} ({total_tfm} aciertos totales)")
    print(f"Frecuencia 100 promedio Top-6: {prom_freq:.4f} ({total_freq} aciertos totales)")
    print(f"Referencia aleatoria teorica Top-6: {esperado:.4f}")
    print(f"Delta TimesFM vs azar: {prom_tfm - esperado:+.4f}")
    print(f"Delta TimesFM vs frecuencia: {prom_tfm - prom_freq:+.4f}")
    print(f"p unilateral exacto TimesFM vs azar*: {p_tfm:.6f}")
    print(f"p unilateral exacto frecuencia vs azar*: {p_freq:.6f}")
    print(f"p pareado TimesFM vs frecuencia**: {p_pareado:.6f}")
    print("* Supone sorteos independientes y 6 numeros equiprobables entre 53.")
    print("** Permutacion por cambio de signo; diagnostico estadistico exploratorio.")


if __name__ == "__main__":
    main()
