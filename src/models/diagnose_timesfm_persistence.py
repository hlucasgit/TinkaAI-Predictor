"""Diagnostica si TimesFM esta actuando como predictor de persistencia (lag-1).

Usa backtest_ensemble_scores.csv, que contiene por sorteo evaluado:
- score TimesFM por numero
- indicador de aparicion real

Para cada evaluacion t (excepto la primera disponible) compara:
1) Top-6 TimesFM para t vs numeros que salieron en t-1
2) numeros reales de t vs numeros reales de t-1
3) Top-6 TimesFM para t vs numeros reales de t

Si (1) es muy superior a (2), TimesFM puede estar sobreponderando la
persistencia del ultimo sorteo en vez de aprender una senal prospectiva util.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "backtest_ensemble_scores.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "diagnostico_timesfm_persistencia_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "diagnostico_timesfm_persistencia_resumen.csv"

TOP_K = 6
TOTAL_NUMEROS = 53


def top6(grupo: pd.DataFrame, columna_score: str) -> set[int]:
    g = grupo.sort_values(
        [columna_score, "Numero"],
        ascending=[False, True],
    )
    return set(g.head(TOP_K)["Numero"].astype(int).tolist())


def reales(grupo: pd.DataFrame) -> set[int]:
    return set(
        grupo.loc[grupo["AparecioReal"].astype(int) == 1, "Numero"]
        .astype(int)
        .tolist()
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe {INPUT_FILE}. Ejecute antes backtest_ensemble.py "
            "con la version actualizada."
        )

    df = pd.read_csv(INPUT_FILE)
    requeridas = {
        "NroEvaluacion",
        "FechaReal",
        "Numero",
        "AparecioReal",
        "ScoreTimesFMNorm",
        "ScoreRecienteNorm",
    }
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    evaluaciones = sorted(df["NroEvaluacion"].astype(int).unique().tolist())
    if len(evaluaciones) < 2:
        raise ValueError("Se necesitan al menos dos evaluaciones")

    filas = []

    for pos in range(1, len(evaluaciones)):
        prev_id = evaluaciones[pos - 1]
        cur_id = evaluaciones[pos]

        prev = df.loc[df["NroEvaluacion"] == prev_id].copy()
        cur = df.loc[df["NroEvaluacion"] == cur_id].copy()

        prev_real = reales(prev)
        cur_real = reales(cur)
        pred_tfm = top6(cur, "ScoreTimesFMNorm")
        pred_freq = top6(cur, "ScoreRecienteNorm")

        if len(prev_real) != TOP_K or len(cur_real) != TOP_K:
            raise ValueError("Cada sorteo debe contener exactamente 6 numeros reales")

        scores_prev_real = cur.loc[
            cur["Numero"].astype(int).isin(prev_real), "ScoreTimesFMNorm"
        ].astype(float)
        scores_no_prev = cur.loc[
            ~cur["Numero"].astype(int).isin(prev_real), "ScoreTimesFMNorm"
        ].astype(float)

        filas.append(
            {
                "NroEvaluacion": cur_id,
                "FechaReal": cur["FechaReal"].iloc[0],
                "OverlapTimesFM_vs_SorteoAnterior": len(pred_tfm & prev_real),
                "OverlapFrecuencia_vs_SorteoAnterior": len(pred_freq & prev_real),
                "RepetidosReales_t_vs_tmenos1": len(cur_real & prev_real),
                "AciertosTimesFM_t": len(pred_tfm & cur_real),
                "AciertosFrecuencia_t": len(pred_freq & cur_real),
                "ScoreTimesFMNormPromedio_NumerosPrevios": float(scores_prev_real.mean()),
                "ScoreTimesFMNormPromedio_Resto": float(scores_no_prev.mean()),
                "PredTimesFM": "-".join(f"{n:02d}" for n in sorted(pred_tfm)),
                "RealesPrevios": "-".join(f"{n:02d}" for n in sorted(prev_real)),
                "RealesActuales": "-".join(f"{n:02d}" for n in sorted(cur_real)),
            }
        )

    detalle = pd.DataFrame(filas)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(DETAIL_FILE, index=False)

    esperado_azar = TOP_K * TOP_K / TOTAL_NUMEROS

    prom_overlap_tfm_prev = float(detalle["OverlapTimesFM_vs_SorteoAnterior"].mean())
    prom_overlap_freq_prev = float(detalle["OverlapFrecuencia_vs_SorteoAnterior"].mean())
    prom_repeticion_real = float(detalle["RepetidosReales_t_vs_tmenos1"].mean())
    prom_hit_tfm = float(detalle["AciertosTimesFM_t"].mean())
    prom_hit_freq = float(detalle["AciertosFrecuencia_t"].mean())
    pct_tfm_4plus_prev = float(
        (detalle["OverlapTimesFM_vs_SorteoAnterior"] >= 4).mean()
    )
    score_prev = float(
        detalle["ScoreTimesFMNormPromedio_NumerosPrevios"].mean()
    )
    score_rest = float(
        detalle["ScoreTimesFMNormPromedio_Resto"].mean()
    )

    resumen = pd.DataFrame(
        [
            {
                "EvaluacionesUtilizadas": len(detalle),
                "OverlapEsperadoAzarTop6": esperado_azar,
                "PromedioOverlapTimesFM_vs_SorteoAnterior": prom_overlap_tfm_prev,
                "PromedioOverlapFrecuencia_vs_SorteoAnterior": prom_overlap_freq_prev,
                "PromedioRepetidosRealesEntreSorteos": prom_repeticion_real,
                "PromedioAciertosTimesFM": prom_hit_tfm,
                "PromedioAciertosFrecuencia": prom_hit_freq,
                "PctPrediccionesTimesFMCon4OMasDelSorteoAnterior": pct_tfm_4plus_prev,
                "ScoreTimesFMPromedioNumerosSorteoAnterior": score_prev,
                "ScoreTimesFMPromedioResto": score_rest,
                "DiferenciaScorePreviosVsResto": score_prev - score_rest,
            }
        ]
    )
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("=" * 78)
    print("TinkaAI Predictor - Diagnostico de persistencia TimesFM")
    print("=" * 78)
    print(f"Evaluaciones: {len(detalle)}")
    print(f"Overlap Top-6 esperado por azar: {esperado_azar:.4f}")
    print("-" * 78)
    print(
        "TimesFM Top-6 vs sorteo anterior: "
        f"{prom_overlap_tfm_prev:.4f} numeros en promedio"
    )
    print(
        "Frecuencia Top-6 vs sorteo anterior: "
        f"{prom_overlap_freq_prev:.4f} numeros en promedio"
    )
    print(
        "Repeticion real sorteo t vs t-1: "
        f"{prom_repeticion_real:.4f} numeros en promedio"
    )
    print(
        "Aciertos TimesFM sobre sorteo actual: "
        f"{prom_hit_tfm:.4f}"
    )
    print(
        "Aciertos frecuencia sobre sorteo actual: "
        f"{prom_hit_freq:.4f}"
    )
    print(
        "Predicciones TimesFM con >=4 numeros del sorteo anterior: "
        f"{pct_tfm_4plus_prev:.2%}"
    )
    print(
        "Score TimesFM medio para numeros del sorteo anterior: "
        f"{score_prev:.4f}"
    )
    print(
        "Score TimesFM medio para el resto: "
        f"{score_rest:.4f}"
    )
    print(
        "Diferencia de score previos - resto: "
        f"{score_prev - score_rest:+.4f}"
    )
    print("-" * 78)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(
        "Lectura: si TimesFM selecciona muchos numeros del sorteo anterior "
        "pero los sorteos reales no repiten en esa magnitud, la aparente senal "
        "puede ser principalmente persistencia lag-1."
    )


if __name__ == "__main__":
    main()
