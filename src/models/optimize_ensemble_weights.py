"""Optimiza pesos del Ensemble sin volver a ejecutar TimesFM.

Usa backtest_ensemble_scores.csv, generado por backtest_ensemble.py.
La seleccion de pesos se hace SOLO con la primera parte cronologica del periodo
y se evalua en la segunda parte, para reducir sobreajuste.

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


def top6_hits(grupo: pd.DataFrame, w_tfm: float, w_hist: float, w_recent: float) -> int:
    score = (
        w_tfm * grupo["ScoreTimesFMNorm"].to_numpy(dtype=float)
        + w_hist * grupo["ScoreHistoricoNorm"].to_numpy(dtype=float)
        + w_recent * grupo["ScoreRecienteNorm"].to_numpy(dtype=float)
    )

    # desempate determinista: score desc, numero asc
    numeros = grupo["Numero"].to_numpy(dtype=int)
    order = np.lexsort((numeros, -score))
    top = set(numeros[order[:TOP_K]].tolist())
    reales = set(
        grupo.loc[grupo["AparecioReal"].astype(int) == 1, "Numero"]
        .astype(int)
        .tolist()
    )
    return len(top & reales)


def evaluar(df: pd.DataFrame, evaluaciones: list[int], pesos: tuple[float, float, float]) -> tuple[int, float]:
    w_tfm, w_hist, w_recent = pesos
    total = 0
    for nro in evaluaciones:
        g = df.loc[df["NroEvaluacion"] == nro]
        total += top6_hits(g, w_tfm, w_hist, w_recent)
    return total, total / len(evaluaciones)


def generar_grid(step: float):
    unidades = int(round(1.0 / step))
    if not np.isclose(unidades * step, 1.0, atol=1e-9):
        raise ValueError("--step debe dividir exactamente 1.0; use por ejemplo 0.05, 0.10 o 0.20")

    for a in range(unidades + 1):
        for b in range(unidades - a + 1):
            c = unidades - a - b
            yield (a * step, b * step, c * step)


def parse_args():
    p = argparse.ArgumentParser(description="Optimiza pesos Ensemble con split cronologico")
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
        total_train, prom_train = evaluar(df, train_ids, pesos)
        filas.append(
            {
                "PesoTimesFM": pesos[0],
                "PesoHistorico": pesos[1],
                "PesoReciente": pesos[2],
                "AciertosTrain": total_train,
                "PromedioTrain": prom_train,
            }
        )

    grid = pd.DataFrame(filas)
    grid = grid.sort_values(
        ["PromedioTrain", "PesoTimesFM", "PesoReciente"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    mejor = grid.iloc[0]
    pesos = (
        float(mejor["PesoTimesFM"]),
        float(mejor["PesoHistorico"]),
        float(mejor["PesoReciente"]),
    )

    total_valid, prom_valid = evaluar(df, valid_ids, pesos)
    total_all, prom_all = evaluar(df, evaluaciones, pesos)

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
                "AciertosTrain": int(mejor["AciertosTrain"]),
                "PromedioTrain": float(mejor["PromedioTrain"]),
                "AciertosValidacion": total_valid,
                "PromedioValidacion": prom_valid,
                "AciertosTotal": total_all,
                "PromedioTotal": prom_all,
                "StepGrid": args.step,
            }
        ]
    )

    GRID_FILE.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_FILE, index=False)
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("=" * 76)
    print("TinkaAI Predictor - Optimizacion de pesos Ensemble")
    print("=" * 76)
    print(f"Train:      {len(train_ids)} sorteos | {f_train_ini} a {f_train_fin}")
    print(f"Validacion: {len(valid_ids)} sorteos | {f_valid_ini} a {f_valid_fin}")
    print(f"Grid step: {args.step:.2f} | configuraciones: {len(grid)}")
    print("-" * 76)
    print(
        "Pesos seleccionados SOLO con train: "
        f"TimesFM={pesos[0]:.2f}, Historico={pesos[1]:.2f}, Reciente={pesos[2]:.2f}"
    )
    print(
        f"Train:      {int(mejor['AciertosTrain'])} aciertos | "
        f"promedio={float(mejor['PromedioTrain']):.4f}"
    )
    print(
        f"Validacion: {total_valid} aciertos | promedio={prom_valid:.4f}"
    )
    print(f"Total:      {total_all} aciertos | promedio={prom_all:.4f}")
    print(f"Grid completo: {GRID_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(
        "Importante: el resultado de validacion es mas informativo que el de train. "
        "La validacion externa de 2026 debe seguir siendo la prueba final."
    )


if __name__ == "__main__":
    main()
