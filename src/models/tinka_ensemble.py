"""TinkaAI Predictor - Ensemble de ranking.

Combina el score de TimesFM con estadisticas historicas/recentes del historial
de sorteos. La salida es un ranking 1..53; no produce probabilidades calibradas.

Notas de modelado:
- TimesFM se usa como score relativo, no como probabilidad.
- La frecuencia historica y reciente son baselines empiricos.
- La recencia se calcula y exporta, pero su peso por defecto es 0 porque un
  numero "atrasado" no es, por ese hecho, mas probable en un sorteo independiente.
- Restricciones de paridad/rangos pertenecen a la generacion de combinaciones,
  no al score individual de cada numero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HIST_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
FORECAST_FILE = ROOT / "data" / "processed" / "prediccion_timesfm.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "ranking_ensemble.csv"

TOTAL_NUMEROS = 53
DEFAULT_RECENT_WINDOW = 100

# Pesos iniciales deliberadamente simples; se validan luego por backtesting.
DEFAULT_W_TIMESFM = 0.50
DEFAULT_W_HIST = 0.20
DEFAULT_W_RECENT = 0.30
DEFAULT_W_RECENCY = 0.00


def normalizar_rango(values: np.ndarray) -> np.ndarray:
    """Convierte un vector a score ordinal 0..1 preservando el ranking.

    Se usa ranking percentil en vez de min-max porque TimesFM puede producir
    muchos valores iguales/cercanos a cero y uno o pocos outliers. Min-max
    aplastaria la mayor parte de la senal contra 0.
    """
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Se encontraron valores no finitos al normalizar")
    if len(x) <= 1:
        return np.full_like(x, 0.5, dtype=float)

    ranks = pd.Series(x).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 1.0) / (len(x) - 1.0)


def validar_pesos(w_timesfm: float, w_hist: float, w_recent: float, w_recency: float) -> None:
    pesos = np.array([w_timesfm, w_hist, w_recent, w_recency], dtype=float)
    if (pesos < 0).any():
        raise ValueError("Los pesos no pueden ser negativos")
    if not np.isclose(pesos.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Los pesos deben sumar 1.0; suma actual={pesos.sum():.6f}")


def cargar_historico() -> tuple[pd.DataFrame, list[str]]:
    if not HIST_FILE.exists():
        raise FileNotFoundError(f"No existe: {HIST_FILE}")

    df = pd.read_csv(HIST_FILE)
    if "Fecha" not in df.columns:
        raise ValueError("timesfm_input.csv requiere la columna Fecha")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el historico: {faltantes}")

    matriz = df[columnas].to_numpy(dtype=float)
    if not np.isin(matriz, [0, 1]).all():
        raise ValueError("El historico debe ser binario (0/1)")

    cantidades = matriz.sum(axis=1)
    if not np.all(cantidades == 6):
        malos = int(np.sum(cantidades != 6))
        raise ValueError(f"Hay {malos} sorteos que no contienen exactamente 6 numeros")

    return df, columnas


def cargar_score_timesfm(columnas: list[str]) -> np.ndarray:
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(f"No existe: {FORECAST_FILE}")

    pred = pd.read_csv(FORECAST_FILE)
    faltantes = [c for c in columnas if c not in pred.columns]
    if faltantes:
        raise ValueError(f"Faltan series TimesFM: {faltantes}")

    if "PasoForecast" in pred.columns:
        fila = pred.loc[pred["PasoForecast"] == 1]
        if fila.empty:
            raise ValueError("prediccion_timesfm.csv no contiene PasoForecast=1")
        fila = fila.iloc[0]
    else:
        if pred.empty:
            raise ValueError("prediccion_timesfm.csv esta vacio")
        fila = pred.iloc[0]

    scores = fila[columnas].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("TimesFM contiene scores no finitos")

    return scores


def calcular_recencia(df: pd.DataFrame, columnas: list[str]) -> np.ndarray:
    gaps = []
    n = len(df)

    for c in columnas:
        indices = np.flatnonzero(df[c].to_numpy(dtype=int) == 1)
        if len(indices) == 0:
            gaps.append(float(n))
        else:
            gaps.append(float((n - 1) - int(indices[-1])))

    return np.asarray(gaps, dtype=float)


def generar_ranking(
    df: pd.DataFrame,
    columnas: list[str],
    score_timesfm_raw: np.ndarray,
    recent_window: int,
    w_timesfm: float,
    w_hist: float,
    w_recent: float,
    w_recency: float,
) -> pd.DataFrame:
    if recent_window < 1:
        raise ValueError("recent_window debe ser >= 1")

    hist_raw = df[columnas].mean(axis=0).to_numpy(dtype=float)
    reciente = df.iloc[-min(recent_window, len(df)):]
    recent_raw = reciente[columnas].mean(axis=0).to_numpy(dtype=float)
    recency_raw = calcular_recencia(df, columnas)

    score_timesfm = normalizar_rango(score_timesfm_raw)
    score_hist = normalizar_rango(hist_raw)
    score_recent = normalizar_rango(recent_raw)
    score_recency = normalizar_rango(recency_raw)

    final = (
        w_timesfm * score_timesfm
        + w_hist * score_hist
        + w_recent * score_recent
        + w_recency * score_recency
    )

    ranking = pd.DataFrame(
        {
            "Numero": np.arange(1, TOTAL_NUMEROS + 1, dtype=int),
            "ScoreFinal": final,
            "ScoreTimesFM": score_timesfm,
            "ScoreFrecuenciaHistorica": score_hist,
            "ScoreFrecuenciaReciente": score_recent,
            "ScoreRecencia": score_recency,
            "ForecastTimesFMRaw": score_timesfm_raw,
            "FrecuenciaHistoricaRaw": hist_raw,
            "FrecuenciaRecienteRaw": recent_raw,
            "SorteosDesdeUltimaAparicion": recency_raw.astype(int),
        }
    )

    ranking = ranking.sort_values(
        ["ScoreFinal", "ScoreTimesFM", "Numero"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Posicion", np.arange(1, len(ranking) + 1))

    return ranking


def parse_args():
    parser = argparse.ArgumentParser(description="Genera ranking Ensemble TinkaAI")
    parser.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW)
    parser.add_argument("--w-timesfm", type=float, default=DEFAULT_W_TIMESFM)
    parser.add_argument("--w-hist", type=float, default=DEFAULT_W_HIST)
    parser.add_argument("--w-recent", type=float, default=DEFAULT_W_RECENT)
    parser.add_argument("--w-recency", type=float, default=DEFAULT_W_RECENCY)
    return parser.parse_args()


def main():
    args = parse_args()
    validar_pesos(args.w_timesfm, args.w_hist, args.w_recent, args.w_recency)

    print("=" * 72)
    print("TinkaAI Predictor - Ensemble v2")
    print("=" * 72)

    df, columnas = cargar_historico()
    score_timesfm_raw = cargar_score_timesfm(columnas)

    ranking = generar_ranking(
        df=df,
        columnas=columnas,
        score_timesfm_raw=score_timesfm_raw,
        recent_window=args.recent_window,
        w_timesfm=args.w_timesfm,
        w_hist=args.w_hist,
        w_recent=args.w_recent,
        w_recency=args.w_recency,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_FILE, index=False)

    print(f"Sorteos cargados: {len(df)}")
    print(f"Ultima fecha historica: {df['Fecha'].max().date()}")
    print(f"Ventana frecuencia reciente: {args.recent_window}")
    print(
        "Diagnostico TimesFM paso 1: "
        f"{int(np.sum(score_timesfm_raw > 0))}/{TOTAL_NUMEROS} scores > 0; "
        f"{len(np.unique(np.round(score_timesfm_raw, 12)))} valores unicos"
    )
    print(
        "Pesos: "
        f"TimesFM={args.w_timesfm:.2f}, "
        f"Hist={args.w_hist:.2f}, "
        f"Reciente={args.w_recent:.2f}, "
        f"Recencia={args.w_recency:.2f}"
    )
    print(f"Ranking generado: {OUTPUT_FILE}")
    print("-" * 72)
    print("Top 10 Ensemble:")
    print(
        ranking[
            [
                "Posicion",
                "Numero",
                "ScoreFinal",
                "ScoreTimesFM",
                "ScoreFrecuenciaHistorica",
                "ScoreFrecuenciaReciente",
                "SorteosDesdeUltimaAparicion",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(
        "Nota: ScoreFinal es un score relativo de ranking; "
        "no es una probabilidad de ganar."
    )


if __name__ == "__main__":
    main()
