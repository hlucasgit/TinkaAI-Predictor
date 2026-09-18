"""Backtest de TimesFM con correccion de persistencia lag-1.

Usa los scores ya calculados en backtest_ensemble_scores.csv, por lo que NO
vuelve a ejecutar TimesFM.

Metodo:
1) Para cada sorteo t, marca si cada numero aparecio en t-1.
2) En TRAIN estima el sesgo medio de TimesFM hacia numeros de t-1:
      alpha = mean(score_tfm | lag1=1) - mean(score_tfm | lag1=0)
3) Define:
      score_debiased = score_tfm - alpha * I(numero aparecio en t-1)
4) Congela alpha y evalua en VALIDACION.

Tambien compara:
- TimesFM original
- TimesFM corregido
- baseline lag-1 (repetir los 6 numeros previos)
- frecuencia reciente

IMPORTANTE: alpha se estima SOLO con TRAIN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "backtest_ensemble_scores.csv"
DETAIL_FILE = ROOT / "data" / "processed" / "backtest_timesfm_debiased_detalle.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "backtest_timesfm_debiased_resumen.csv"

TOP_K = 6


def top6(grupo: pd.DataFrame, score_col: str) -> set[int]:
    g = grupo.sort_values([score_col, "Numero"], ascending=[False, True])
    return set(g.head(TOP_K)["Numero"].astype(int).tolist())


def reales(grupo: pd.DataFrame) -> set[int]:
    return set(
        grupo.loc[grupo["AparecioReal"].astype(int) == 1, "Numero"]
        .astype(int)
        .tolist()
    )


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    evaluaciones = sorted(df["NroEvaluacion"].astype(int).unique().tolist())
    filas = []

    for pos in range(1, len(evaluaciones)):
        prev_id = evaluaciones[pos - 1]
        cur_id = evaluaciones[pos]

        prev = df.loc[df["NroEvaluacion"] == prev_id]
        cur = df.loc[df["NroEvaluacion"] == cur_id].copy()

        prev_real = reales(prev)
        cur["Lag1"] = cur["Numero"].astype(int).isin(prev_real).astype(int)
        filas.append(cur)

    if not filas:
        raise ValueError("No hay evaluaciones suficientes")

    return pd.concat(filas, ignore_index=True)


def estimar_alpha(train: pd.DataFrame) -> tuple[float, float, float]:
    s1 = train.loc[train["Lag1"] == 1, "ScoreTimesFMNorm"].astype(float)
    s0 = train.loc[train["Lag1"] == 0, "ScoreTimesFMNorm"].astype(float)

    if s1.empty or s0.empty:
        raise ValueError("No hay datos suficientes para estimar alpha")

    m1 = float(s1.mean())
    m0 = float(s0.mean())
    return m1 - m0, m1, m0


def evaluar_bloque(df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
    filas = []

    for nro in sorted(df["NroEvaluacion"].astype(int).unique().tolist()):
        g = df.loc[df["NroEvaluacion"] == nro].copy()
        g["ScoreTimesFMDebiased"] = (
            g["ScoreTimesFMNorm"].astype(float)
            - alpha * g["Lag1"].astype(float)
        )

        real = reales(g)
        pred_tfm = top6(g, "ScoreTimesFMNorm")
        pred_deb = top6(g, "ScoreTimesFMDebiased")
        pred_freq = top6(g, "ScoreRecienteNorm")
        pred_lag1 = set(
            g.loc[g["Lag1"] == 1, "Numero"].astype(int).tolist()
        )

        filas.append(
            {
                "NroEvaluacion": nro,
                "FechaReal": g["FechaReal"].iloc[0],
                "AciertosTimesFM": len(pred_tfm & real),
                "AciertosTimesFMDebiased": len(pred_deb & real),
                "AciertosFrecuencia": len(pred_freq & real),
                "AciertosLag1": len(pred_lag1 & real),
                "PredTimesFM": "-".join(f"{n:02d}" for n in sorted(pred_tfm)),
                "PredTimesFMDebiased": "-".join(f"{n:02d}" for n in sorted(pred_deb)),
                "PredFrecuencia": "-".join(f"{n:02d}" for n in sorted(pred_freq)),
                "PredLag1": "-".join(f"{n:02d}" for n in sorted(pred_lag1)),
                "NumerosReales": "-".join(f"{n:02d}" for n in sorted(real)),
            }
        )

    det = pd.DataFrame(filas)
    stats = {
        "n": len(det),
        "tfm_total": int(det["AciertosTimesFM"].sum()),
        "tfm_prom": float(det["AciertosTimesFM"].mean()),
        "deb_total": int(det["AciertosTimesFMDebiased"].sum()),
        "deb_prom": float(det["AciertosTimesFMDebiased"].mean()),
        "freq_total": int(det["AciertosFrecuencia"].sum()),
        "freq_prom": float(det["AciertosFrecuencia"].mean()),
        "lag1_total": int(det["AciertosLag1"].sum()),
        "lag1_prom": float(det["AciertosLag1"].mean()),
    }
    return det, stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Evalua TimesFM corregido por persistencia lag-1"
    )
    p.add_argument("--train-frac", type=float, default=0.50)
    return p.parse_args()


def main():
    args = parse_args()

    if not 0.2 <= args.train_frac <= 0.8:
        raise ValueError("--train-frac debe estar entre 0.2 y 0.8")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

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

    data = preparar(df)
    evaluaciones = sorted(data["NroEvaluacion"].astype(int).unique().tolist())
    corte = int(len(evaluaciones) * args.train_frac)

    train_ids = evaluaciones[:corte]
    valid_ids = evaluaciones[corte:]

    train = data.loc[data["NroEvaluacion"].isin(train_ids)].copy()
    valid = data.loc[data["NroEvaluacion"].isin(valid_ids)].copy()

    alpha, score_lag1, score_resto = estimar_alpha(train)

    det_train, st_train = evaluar_bloque(train, alpha)
    det_train.insert(0, "Bloque", "TRAIN")

    det_valid, st_valid = evaluar_bloque(valid, alpha)
    det_valid.insert(0, "Bloque", "VALIDACION")

    detalle = pd.concat([det_train, det_valid], ignore_index=True)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(DETAIL_FILE, index=False)

    train_ini = det_train["FechaReal"].iloc[0]
    train_fin = det_train["FechaReal"].iloc[-1]
    valid_ini = det_valid["FechaReal"].iloc[0]
    valid_fin = det_valid["FechaReal"].iloc[-1]

    resumen = pd.DataFrame(
        [
            {
                "TrainSorteos": st_train["n"],
                "TrainInicio": train_ini,
                "TrainFin": train_fin,
                "ValidacionSorteos": st_valid["n"],
                "ValidacionInicio": valid_ini,
                "ValidacionFin": valid_fin,
                "AlphaPersistenciaEstimadoEnTrain": alpha,
                "ScoreTimesFMPromedioLag1EnTrain": score_lag1,
                "ScoreTimesFMPromedioRestoEnTrain": score_resto,
                "TrainPromedioTimesFM": st_train["tfm_prom"],
                "TrainPromedioTimesFMDebiased": st_train["deb_prom"],
                "TrainPromedioFrecuencia": st_train["freq_prom"],
                "TrainPromedioLag1": st_train["lag1_prom"],
                "ValidacionPromedioTimesFM": st_valid["tfm_prom"],
                "ValidacionPromedioTimesFMDebiased": st_valid["deb_prom"],
                "ValidacionPromedioFrecuencia": st_valid["freq_prom"],
                "ValidacionPromedioLag1": st_valid["lag1_prom"],
                "DeltaDebiasedVsTimesFMValidacion": (
                    st_valid["deb_prom"] - st_valid["tfm_prom"]
                ),
                "DeltaDebiasedVsFrecuenciaValidacion": (
                    st_valid["deb_prom"] - st_valid["freq_prom"]
                ),
            }
        ]
    )
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("=" * 78)
    print("TinkaAI Predictor - TimesFM corregido por persistencia lag-1")
    print("=" * 78)
    print(
        f"Train:      {st_train['n']} sorteos | {train_ini} a {train_fin}"
    )
    print(
        f"Validacion: {st_valid['n']} sorteos | {valid_ini} a {valid_fin}"
    )
    print("-" * 78)
    print(f"Score TimesFM medio lag-1 en TRAIN: {score_lag1:.4f}")
    print(f"Score TimesFM medio resto en TRAIN: {score_resto:.4f}")
    print(f"Alpha congelado: {alpha:.4f}")
    print("-" * 78)
    print("TRAIN")
    print(
        f"TimesFM:          {st_train['tfm_prom']:.4f} "
        f"({st_train['tfm_total']})"
    )
    print(
        f"TimesFM debiased: {st_train['deb_prom']:.4f} "
        f"({st_train['deb_total']})"
    )
    print(
        f"Frecuencia:       {st_train['freq_prom']:.4f} "
        f"({st_train['freq_total']})"
    )
    print(
        f"Lag-1 baseline:   {st_train['lag1_prom']:.4f} "
        f"({st_train['lag1_total']})"
    )
    print("-" * 78)
    print("VALIDACION")
    print(
        f"TimesFM:          {st_valid['tfm_prom']:.4f} "
        f"({st_valid['tfm_total']})"
    )
    print(
        f"TimesFM debiased: {st_valid['deb_prom']:.4f} "
        f"({st_valid['deb_total']})"
    )
    print(
        f"Frecuencia:       {st_valid['freq_prom']:.4f} "
        f"({st_valid['freq_total']})"
    )
    print(
        f"Lag-1 baseline:   {st_valid['lag1_prom']:.4f} "
        f"({st_valid['lag1_total']})"
    )
    print(
        "Delta debiased vs TimesFM en validacion: "
        f"{st_valid['deb_prom'] - st_valid['tfm_prom']:+.4f}"
    )
    print(
        "Delta debiased vs frecuencia en validacion: "
        f"{st_valid['deb_prom'] - st_valid['freq_prom']:+.4f}"
    )
    print("-" * 78)
    print(f"Detalle: {DETAIL_FILE}")
    print(f"Resumen: {SUMMARY_FILE}")
    print(
        "Interpretacion: si la correccion mejora validacion, parte del problema "
        "era el sesgo lag-1. Si no mejora, conviene retirar TimesFM de la señal "
        "operativa y conservarlo solo como experimento."
    )


if __name__ == "__main__":
    main()
