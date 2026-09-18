"""Prediccion operativa TinkaAI usando todo el historial disponible.

Construye un historial actualizado combinando:
- data/processed/timesfm_input.csv (base historica)
- data/raw/tinka_actualizacion_2026.csv (sorteos posteriores al cutoff)

Luego:
1) carga los pesos congelados por optimize_ensemble_weights.py,
2) calcula TimesFM (un paso), frecuencia historica y frecuencia reciente,
3) genera ranking 1..53,
4) genera combinaciones diversificadas.

IMPORTANTE: los scores son experimentales y NO son probabilidades calibradas.
En una loteria justa, ninguna combinacion tiene garantia de ser mas probable.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm


ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
UPDATE_FILE = ROOT / "data" / "raw" / "tinka_actualizacion_2026.csv"
WEIGHTS_FILE = ROOT / "data" / "processed" / "optimizacion_pesos_ensemble_resumen.csv"

CURRENT_INPUT_FILE = ROOT / "data" / "processed" / "timesfm_input_actual.csv"
RANKING_FILE = ROOT / "data" / "processed" / "ranking_actual.csv"
COMBOS_FILE = ROOT / "data" / "processed" / "combinaciones_actual.csv"

TOTAL_NUMEROS = 53
TOP_K = 6
MAX_CONTEXT = 2048
DEFAULT_RECENT_WINDOW = 100
DEFAULT_POOL = 18
DEFAULT_JUEGOS = 10
DEFAULT_MAX_COMUNES = 4


def normalizar_rango(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Se encontraron valores no finitos")
    if len(x) <= 1:
        return np.full_like(x, 0.5, dtype=float)
    ranks = pd.Series(x).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 1.0) / (len(x) - 1.0)


def cargar_base() -> tuple[pd.DataFrame, list[str]]:
    if not BASE_FILE.exists():
        raise FileNotFoundError(BASE_FILE)

    df = pd.read_csv(BASE_FILE)
    if "Fecha" not in df.columns:
        raise ValueError("timesfm_input.csv requiere Fecha")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas base: {faltan}")

    return df[["Fecha"] + columnas].sort_values("Fecha").reset_index(drop=True), columnas


def cargar_actualizaciones(columnas: list[str]) -> pd.DataFrame:
    if not UPDATE_FILE.exists():
        raise FileNotFoundError(UPDATE_FILE)

    raw = pd.read_csv(UPDATE_FILE)
    req = ["Fecha"] + [f"Numero{i}" for i in range(1, 7)]
    faltan = [c for c in req if c not in raw.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en actualizacion: {faltan}")

    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    filas = []

    for _, r in raw.iterrows():
        nums = [int(r[f"Numero{i}"]) for i in range(1, 7)]
        if len(set(nums)) != TOP_K:
            raise ValueError(f"Numeros repetidos en fecha {r['Fecha']}")
        if min(nums) < 1 or max(nums) > TOTAL_NUMEROS:
            raise ValueError(f"Numero fuera de 1..53 en fecha {r['Fecha']}")

        fila = {"Fecha": r["Fecha"]}
        for c in columnas:
            fila[c] = 0
        for n in nums:
            fila[f"N{n:02d}"] = 1
        filas.append(fila)

    return pd.DataFrame(filas).sort_values("Fecha").reset_index(drop=True)


def construir_historial_actual() -> tuple[pd.DataFrame, list[str]]:
    base, columnas = cargar_base()
    updates = cargar_actualizaciones(columnas)

    # Las actualizaciones prevalecen si por alguna razon una fecha ya existiera.
    fechas_update = set(updates["Fecha"].tolist())
    base = base.loc[~base["Fecha"].isin(fechas_update)]

    df = pd.concat([base, updates], ignore_index=True)
    df = df.sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True)

    matriz = df[columnas].to_numpy(dtype=int)
    if not np.isin(matriz, [0, 1]).all():
        raise ValueError("Historial actual debe ser binario")
    cantidades = matriz.sum(axis=1)
    if not np.all(cantidades == TOP_K):
        n_malos = int((cantidades != TOP_K).sum())
        raise ValueError(f"Hay {n_malos} filas que no contienen exactamente 6 numeros")

    CURRENT_INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CURRENT_INPUT_FILE, index=False)
    return df, columnas


def cargar_pesos() -> tuple[float, float, float]:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"No existe {WEIGHTS_FILE}. Ejecute primero optimize_ensemble_weights.py"
        )

    r = pd.read_csv(WEIGHTS_FILE)
    if r.empty:
        raise ValueError("Archivo de pesos vacio")

    fila = r.iloc[0]
    pesos = (
        float(fila["PesoTimesFM"]),
        float(fila["PesoHistorico"]),
        float(fila["PesoReciente"]),
    )
    if any(p < 0 for p in pesos) or not np.isclose(sum(pesos), 1.0, atol=1e-9):
        raise ValueError(f"Pesos invalidos: {pesos}")
    return pesos


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


def forecast_timesfm(
    df: pd.DataFrame,
    columnas: list[str],
) -> np.ndarray:
    contexto = df.iloc[-min(MAX_CONTEXT, len(df)):]
    series = [contexto[c].to_numpy(dtype=np.float32) for c in columnas]
    model = cargar_modelo()
    point, _ = model.forecast(horizon=1, inputs=series)
    scores = np.asarray(point, dtype=float)[:, 0]
    if len(scores) != TOTAL_NUMEROS:
        raise ValueError("TimesFM no devolvio 53 scores")
    return scores


def generar_ranking(
    df: pd.DataFrame,
    columnas: list[str],
    tfm_raw: np.ndarray,
    recent_window: int,
    pesos: tuple[float, float, float],
) -> pd.DataFrame:
    w_tfm, w_hist, w_recent = pesos

    hist_raw = df[columnas].mean(axis=0).to_numpy(dtype=float)
    recent_raw = (
        df.iloc[-min(recent_window, len(df)):][columnas]
        .mean(axis=0)
        .to_numpy(dtype=float)
    )

    tfm_norm = normalizar_rango(tfm_raw)
    hist_norm = normalizar_rango(hist_raw)
    recent_norm = normalizar_rango(recent_raw)

    final = w_tfm * tfm_norm + w_hist * hist_norm + w_recent * recent_norm

    out = pd.DataFrame(
        {
            "Numero": np.arange(1, TOTAL_NUMEROS + 1, dtype=int),
            "ScoreFinal": final,
            "ScoreTimesFMNorm": tfm_norm,
            "ScoreHistoricoNorm": hist_norm,
            "ScoreRecienteNorm": recent_norm,
            "TimesFMRaw": tfm_raw,
            "FrecuenciaHistoricaRaw": hist_raw,
            "FrecuenciaRecienteRaw": recent_raw,
        }
    )

    out = out.sort_values(
        ["ScoreFinal", "ScoreTimesFMNorm", "Numero"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "Posicion", np.arange(1, len(out) + 1))
    return out


def historicas_set(df: pd.DataFrame, columnas: list[str]) -> set[tuple[int, ...]]:
    out: set[tuple[int, ...]] = set()
    for _, fila in df.iterrows():
        nums = tuple(
            i for i in range(1, TOTAL_NUMEROS + 1)
            if int(fila[f"N{i:02d}"]) == 1
        )
        if len(nums) == TOP_K:
            out.add(nums)
    return out


def stats_combo(combo: tuple[int, ...]) -> dict[str, int]:
    impares = sum(n % 2 != 0 for n in combo)
    bajos = sum(1 <= n <= 17 for n in combo)
    medios = sum(18 <= n <= 35 for n in combo)
    altos = sum(36 <= n <= 53 for n in combo)

    max_consecutivos = 1
    actual = 1
    for a, b in zip(combo, combo[1:]):
        if b == a + 1:
            actual += 1
            max_consecutivos = max(max_consecutivos, actual)
        else:
            actual = 1

    return {
        "Pares": TOP_K - impares,
        "Impares": impares,
        "Bajos01_17": bajos,
        "Medios18_35": medios,
        "Altos36_53": altos,
        "MaxConsecutivos": max_consecutivos,
    }


def combo_valida(combo: tuple[int, ...]) -> bool:
    s = stats_combo(combo)
    if s["Impares"] < 2 or s["Impares"] > 4:
        return False
    if min(s["Bajos01_17"], s["Medios18_35"], s["Altos36_53"]) == 0:
        return False
    if max(s["Bajos01_17"], s["Medios18_35"], s["Altos36_53"]) > 3:
        return False
    if s["MaxConsecutivos"] > 3:
        return False
    return True


def generar_combinaciones(
    ranking: pd.DataFrame,
    historial: pd.DataFrame,
    columnas: list[str],
    pool: int,
    juegos: int,
    max_comunes: int,
) -> pd.DataFrame:
    pool_nums = sorted(ranking.head(pool)["Numero"].astype(int).tolist())
    score_map = dict(
        zip(ranking["Numero"].astype(int), ranking["ScoreFinal"].astype(float))
    )
    historicas = historicas_set(historial, columnas)

    candidatas = []
    for combo in itertools.combinations(pool_nums, TOP_K):
        if combo in historicas or not combo_valida(combo):
            continue
        scores = [score_map[n] for n in combo]
        candidatas.append((combo, float(np.mean(scores)), float(np.min(scores))))

    candidatas.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)

    seleccionadas = []
    for cand in candidatas:
        combo = cand[0]
        if all(len(set(combo) & set(prev[0])) <= max_comunes for prev in seleccionadas):
            seleccionadas.append(cand)
            if len(seleccionadas) >= juegos:
                break

    filas = []
    for i, (combo, promedio, minimo) in enumerate(seleccionadas, start=1):
        filas.append(
            {
                "Juego": i,
                **{f"Numero{j}": n for j, n in enumerate(combo, start=1)},
                "ScorePromedio": promedio,
                "ScoreMinimo": minimo,
                **stats_combo(combo),
            }
        )
    return pd.DataFrame(filas)


def parse_args():
    p = argparse.ArgumentParser(description="Genera ranking y combinaciones actuales")
    p.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW)
    p.add_argument("--pool", type=int, default=DEFAULT_POOL)
    p.add_argument("--juegos", type=int, default=DEFAULT_JUEGOS)
    p.add_argument("--max-comunes", type=int, default=DEFAULT_MAX_COMUNES)
    return p.parse_args()


def main():
    args = parse_args()

    if args.recent_window < 1:
        raise ValueError("--recent-window debe ser >= 1")
    if not TOP_K <= args.pool <= TOTAL_NUMEROS:
        raise ValueError("--pool debe estar entre 6 y 53")
    if args.juegos < 1:
        raise ValueError("--juegos debe ser >= 1")
    if not 0 <= args.max_comunes < TOP_K:
        raise ValueError("--max-comunes debe estar entre 0 y 5")

    print("=" * 78)
    print("TinkaAI Predictor - Prediccion actual")
    print("=" * 78)

    historial, columnas = construir_historial_actual()
    pesos = cargar_pesos()

    print(f"Sorteos disponibles: {len(historial)}")
    print(f"Ultima fecha incorporada: {historial['Fecha'].max().date()}")
    print(
        "Pesos congelados: "
        f"TimesFM={pesos[0]:.2f}, Historico={pesos[1]:.2f}, Reciente={pesos[2]:.2f}"
    )
    print("Calculando TimesFM un paso...")

    tfm_raw = forecast_timesfm(historial, columnas)

    ranking = generar_ranking(
        historial,
        columnas,
        tfm_raw,
        args.recent_window,
        pesos,
    )
    ranking.to_csv(RANKING_FILE, index=False)

    combos = generar_combinaciones(
        ranking,
        historial,
        columnas,
        args.pool,
        args.juegos,
        args.max_comunes,
    )
    combos.to_csv(COMBOS_FILE, index=False)

    print("-" * 78)
    print(f"Historial actual: {CURRENT_INPUT_FILE}")
    print(f"Ranking actual: {RANKING_FILE}")
    print(f"Combinaciones: {COMBOS_FILE}")
    print("-" * 78)
    print("Top 10 actual:")
    print(
        ranking[
            [
                "Posicion",
                "Numero",
                "ScoreFinal",
                "ScoreTimesFMNorm",
                "ScoreHistoricoNorm",
                "ScoreRecienteNorm",
            ]
        ].head(10).to_string(index=False)
    )

    print("-" * 78)
    print("Combinaciones candidatas:")
    for _, r in combos.iterrows():
        nums = "-".join(f"{int(r[f'Numero{i}']):02d}" for i in range(1, 7))
        print(f"Juego {int(r['Juego']):02d}: {nums} | score={r['ScorePromedio']:.4f}")

    print(
        "Nota: estas combinaciones son una salida experimental de ranking y cobertura; "
        "no existe garantia de que sean mas probables en un sorteo justo."
    )


if __name__ == "__main__":
    main()
