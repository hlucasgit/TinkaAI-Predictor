"""Optimiza pesos del Ensemble sin volver a ejecutar TimesFM.

Usa backtest_ensemble_scores.csv, generado por backtest_ensemble.py.

La seleccion de pesos se hace SOLO con la primera parte cronologica del periodo
y se evalua en la segunda parte. El criterio principal es "acierto esperado con
empates", no un desempate arbitrario por numero. Esto es importante porque las
frecuencias suelen producir muchos scores iguales alrededor del corte Top-6.

Por defecto se optimizan:
- TimesFM
- Frecuencia historica
- Frecuencia reciente

Recencia queda en 0 por defecto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "backtest_ensemble_scores.csv"
GRID_FILE = ROOT / "data" / "processed" / "optimizacion_pesos_ensemble_grid.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "optimizacion_pesos_ensemble_resumen.csv"

TOP_K = 6
TIE_TOL = 1e-12


def calcular_score(
    grupo: pd.DataFrame,
    w_tfm: float,
    w_hist: float,
    w_recent: float,
) -> np.ndarray:
    return (
        w_tfm * grupo["ScoreTimesFMNorm"].to_numpy(dtype=float)
        + w_hist * grupo["ScoreHistoricoNorm"].to_numpy(dtype=float)
        + w_recent * grupo["ScoreRecienteNorm"].to_numpy(dtype=float)
    )


def hits_deterministas(
    grupo: pd.DataFrame,
    score: np.ndarray,
) -> int:
    """Top-6 reproducible; score desc y numero asc solo para desempatar."""
    numeros = grupo["Numero"].to_numpy(dtype=int)
    order = np.lexsort((numeros, -score))
    seleccion = order[:TOP_K]
    real = grupo["AparecioReal"].to_numpy(dtype=int)
    return int(real[seleccion].sum())


def hits_esperados_con_empates(
    grupo: pd.DataFrame,
    score: np.ndarray,
) -> float:
    """Aciertos esperados si el empate en el limite Top-6 se rompe al azar.

    Evita que una configuracion gane el grid solo porque, por ejemplo, ante una
    frecuencia empatada se eligieron numeros bajos o altos de forma arbitraria.
    """
    real = grupo["AparecioReal"].to_numpy(dtype=int)

    # sexto score mas alto
    threshold = float(np.sort(score)[-TOP_K])
    superiores = score > threshold + TIE_TOL
    empatados = np.isclose(score, threshold, rtol=0.0, atol=TIE_TOL)

    n_superiores = int(superiores.sum())
    espacios = TOP_K - n_superiores
    if espacios < 0:
        raise RuntimeError("Inconsistencia al calcular el corte Top-6")

    hits_superiores = float(real[superiores].sum())
    n_empatados = int(empatados.sum())

    if espacios == 0:
        return hits_superiores
    if n_empatados == 0:
        raise RuntimeError("No se encontro grupo de empate en el corte Top-6")

    hits_empatados = float(real[empatados].sum())
    return hits_superiores + espacios * (hits_empatados / n_empatados)


def evaluar(
    df: pd.DataFrame,
    evaluaciones: list[int],
    pesos: tuple[float, float, float],
) -> tuple[float, float, int, float]:
    w_tfm, w_hist, w_recent = pesos
    total_esperado = 0.0
    total_determinista = 0

    for nro in evaluaciones:
        g = df.loc[df["NroEvaluacion"] == nro]
        score = calcular_score(g, w_tfm, w_hist, w_recent)
        total_esperado += hits_esperados_con_empates(g, score)
        total_determinista += hits_deterministas(g, score)

    n = len(evaluaciones)
    return (
        total_esperado,
        total_esperado / n,
        total_determinista,
        total_determinista / n,
    )


def generar_grid(step: float):
    unidades = int(round(1.0 / step))
    if not np.isclose(unidades * step, 1.0, atol=1e-9):
        raise ValueError(
            "--step debe dividir exactamente 1.0; use por ejemplo 0.05, 0.10 o 0.20"
        )

    for a in range(unidades + 1):
        for b in range(unidades - a + 1):
            c = unidades - a - b
            yield (a * step, b * step, c * step)


def parse_args():
    p = argparse.ArgumentParser(
        description="Optimiza pesos Ensemble con split cronologico y empates robustos"
    )
    p.add_argument("--train-frac", type=float, default=0.50)
    p.add_argument("--step", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()

    if not 0.2 <= args.train_frac <= 0.8:
        raise ValueError("--train-frac debe estar entre 0.2 y 0.8")
    if not 0 < args.step <= 0.5:
        raise ValueError("--step debe estar entre 0 y 0.5")

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
        "ScoreHistoricoNorm",
        "ScoreRecienteNorm",
    }
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    evaluaciones = sorted(df["NroEvaluacion"].astype(int).unique().tolist())
    if len(evaluaciones) < 20:
        raise ValueError("Se requieren al menos 20 sorteos evaluados")

    corte = int(len(evaluaciones) * args.train_frac)
    train_ids = evaluaciones[:corte]
    valid_ids = evaluaciones[corte:]

    filas = []
    for pesos in generar_grid(args.step):
        exp_total, exp_prom, det_total, det_prom = evaluar(df, train_ids, pesos)
        filas.append(
            {
                "PesoTimesFM": pesos[0],
                "PesoHistorico": pesos[1],
                "PesoReciente": pesos[2],
                "AciertosEsperadosTrain": exp_total,
                "PromedioEsperadoTrain": exp_prom,
                "AciertosDeterministasTrain": det_total,
                "PromedioDeterministaTrain": det_prom,
            }
        )

    grid = pd.DataFrame(filas)
    grid = grid.sort_values(
        [
            "PromedioEsperadoTrain",
            "PromedioDeterministaTrain",
            "PesoTimesFM",
            "PesoReciente",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    mejor = grid.iloc[0]
    pesos = (
        float(mejor["PesoTimesFM"]),
        float(mejor["PesoHistorico"]),
        float(mejor["PesoReciente"]),
    )

    exp_valid, exp_prom_valid, det_valid, det_prom_valid = evaluar(
        df, valid_ids, pesos
    )
    exp_all, exp_prom_all, det_all, det_prom_all = evaluar(
        df, evaluaciones, pesos
    )

    f_train_ini = df.loc[df["NroEvaluacion"] == train_ids[0], "FechaReal"].iloc[0]
    f_train_fin = df.loc[df["NroEvaluacion"] == train_ids[-1], "FechaReal"].iloc[0]
    f_valid_ini = df.loc[df["NroEvaluacion"] == valid_ids[0], "FechaReal"].iloc[0]
    f_valid_fin = df.loc[df["NroEvaluacion"] == valid_ids[-1], "FechaReal"].iloc[0]

    resumen = pd.DataFrame(
        [
            {
                "SorteosTotales": len(evaluaciones),
                "SorteosTrain": len(train_ids),
                "TrainInicio": f_train_ini,
                "TrainFin": f_train_fin,
                "SorteosValidacion": len(valid_ids),
                "ValidacionInicio": f_valid_ini,
                "ValidacionFin": f_valid_fin,
                "PesoTimesFM": pesos[0],
                "PesoHistorico": pesos[1],
                "PesoReciente": pesos[2],
                "PesoRecencia": 0.0,
                "AciertosEsperadosTrain": float(mejor["AciertosEsperadosTrain"]),
                "PromedioEsperadoTrain": float(mejor["PromedioEsperadoTrain"]),
                "AciertosDeterministasTrain": int(mejor["AciertosDeterministasTrain"]),
                "PromedioDeterministaTrain": float(mejor["PromedioDeterministaTrain"]),
                "AciertosEsperadosValidacion": exp_valid,
                "PromedioEsperadoValidacion": exp_prom_valid,
                "AciertosDeterministasValidacion": det_valid,
                "PromedioDeterministaValidacion": det_prom_valid,
                "AciertosEsperadosTotal": exp_all,
                "PromedioEsperadoTotal": exp_prom_all,
                "AciertosDeterministasTotal": det_all,
                "PromedioDeterministaTotal": det_prom_all,
                "StepGrid": args.step,
            }
        ]
    )

    GRID_FILE.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_FILE, index=False)
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("=" * 78)
    print("TinkaAI Predictor - Optimizacion de pesos Ensemble (tie-aware)")
    print("=" * 78)
    print(f"Train:      {len(train_ids)} sorteos | {f_train_ini} a {f_train_fin}")
    print(f"Validacion: {len(valid_ids)} sorteos | {f_valid_ini} a {f_valid_fin}")
    print(f"Grid step: {args.step:.2f} | configuraciones: {len(grid)}")
    print("-" * 78)
    print(
        "Pesos seleccionados SOLO con train: "
        f"TimesFM={pesos[0]:.2f}, Historico={pesos[1]:.2f}, "
        f"Reciente={pesos[2]:.2f}"
    )
    print(
        "Train tie-aware:      "
        f"{float(mejor['AciertosEsperadosTrain']):.3f} | "
        f"promedio={float(mejor['PromedioEsperadoTrain']):.4f}"
    )
    print(
        "Train determinista:    "
        f"{int(mejor['AciertosDeterministasTrain'])} | "
        f"promedio={float(mejor['PromedioDeterministaTrain']):.4f}"
    )
    print(
        f"Validacion tie-aware: {exp_valid:.3f} | promedio={exp_prom_valid:.4f}"
    )
    print(
        f"Validacion determinista: {det_valid} | promedio={det_prom_valid:.4f}"
    )
    print(f"Total tie-aware:       {exp_all:.3f} | promedio={exp_prom_all:.4f}")
    print(f"Total determinista:    {det_all} | promedio={det_prom_all:.4f}")
    print("-" * 78)
    print("Top 10 configuraciones segun TRAIN:")
    print(
        grid[
            [
                "PesoTimesFM",
                "PesoHistorico",
                "PesoReciente",
                "PromedioEsperadoTrain",
                "PromedioDeterministaTrain",
            ]
        ].head(10).to_string(index=False)
    )
    print(f"Grid completo: {GRID_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(
        "Nota: tie-aware estima el resultado esperado cuando varios numeros "
        "empatan en el sexto puesto. Evita premiar un desempate numerico arbitrario."
    )


if __name__ == "__main__":
    main()
