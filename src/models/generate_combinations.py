"""Genera combinaciones candidatas desde ranking_ensemble.csv.

Este modulo no aumenta la probabilidad matematica de una combinacion. Su funcion
es convertir un ranking de numeros en un conjunto pequeno y diverso de jugadas,
aplicando restricciones estructurales y evitando repetir sorteos historicos.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RANKING_FILE = ROOT / "data" / "processed" / "ranking_ensemble.csv"
HIST_FILE = ROOT / "data" / "processed" / "timesfm_input.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "combinaciones_ensemble.csv"

TOTAL_NUMEROS = 53
TOP_K = 6
DEFAULT_POOL = 18
DEFAULT_JUEGOS = 10
DEFAULT_MAX_COMUNES = 4


def cargar_ranking() -> pd.DataFrame:
    if not RANKING_FILE.exists():
        raise FileNotFoundError(f"No existe: {RANKING_FILE}")

    df = pd.read_csv(RANKING_FILE)
    requeridas = ["Numero", "ScoreFinal"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en ranking_ensemble.csv: {faltantes}")

    df["Numero"] = df["Numero"].astype(int)
    if len(df) != TOTAL_NUMEROS or df["Numero"].nunique() != TOTAL_NUMEROS:
        raise ValueError("ranking_ensemble.csv debe contener exactamente los numeros 1..53")

    return df.sort_values("ScoreFinal", ascending=False).reset_index(drop=True)


def cargar_historicas() -> set[tuple[int, ...]]:
    if not HIST_FILE.exists():
        raise FileNotFoundError(f"No existe: {HIST_FILE}")

    df = pd.read_csv(HIST_FILE)
    columnas = [f"N{i:02d}" for i in range(1, TOTAL_NUMEROS + 1)]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en timesfm_input.csv: {faltantes}")

    historicas: set[tuple[int, ...]] = set()
    for _, fila in df.iterrows():
        nums = tuple(i for i in range(1, TOTAL_NUMEROS + 1) if int(fila[f"N{i:02d}"]) == 1)
        if len(nums) == TOP_K:
            historicas.add(nums)
    return historicas


def estadisticas_combo(combo: tuple[int, ...]) -> dict[str, int]:
    impares = sum(n % 2 != 0 for n in combo)
    pares = TOP_K - impares
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
        "Pares": pares,
        "Impares": impares,
        "Bajos01_17": bajos,
        "Medios18_35": medios,
        "Altos36_53": altos,
        "MaxConsecutivos": max_consecutivos,
    }


def es_valida(combo: tuple[int, ...]) -> bool:
    e = estadisticas_combo(combo)

    # Filtros de diversidad estructural, no reglas predictivas.
    if e["Impares"] < 2 or e["Impares"] > 4:
        return False
    if e["Bajos01_17"] == 0 or e["Medios18_35"] == 0 or e["Altos36_53"] == 0:
        return False
    if max(e["Bajos01_17"], e["Medios18_35"], e["Altos36_53"]) > 3:
        return False
    if e["MaxConsecutivos"] > 3:
        return False

    return True


def score_combo(combo: tuple[int, ...], score_por_numero: dict[int, float]) -> tuple[float, float]:
    scores = [float(score_por_numero[n]) for n in combo]
    return sum(scores) / len(scores), min(scores)


def seleccionar_diversas(
    candidatas: list[tuple[tuple[int, ...], float, float]],
    cantidad: int,
    max_comunes: int,
) -> list[tuple[tuple[int, ...], float, float]]:
    seleccionadas = []

    for cand in candidatas:
        combo = cand[0]
        if all(len(set(combo) & set(prev[0])) <= max_comunes for prev in seleccionadas):
            seleccionadas.append(cand)
            if len(seleccionadas) >= cantidad:
                break

    return seleccionadas


def parse_args():
    p = argparse.ArgumentParser(description="Genera combinaciones desde ranking Ensemble")
    p.add_argument("--pool", type=int, default=DEFAULT_POOL, help="Cantidad de numeros top a combinar")
    p.add_argument("--juegos", type=int, default=DEFAULT_JUEGOS, help="Cantidad de combinaciones a generar")
    p.add_argument(
        "--max-comunes",
        type=int,
        default=DEFAULT_MAX_COMUNES,
        help="Maximo de numeros compartidos entre dos combinaciones seleccionadas",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.pool < TOP_K or args.pool > TOTAL_NUMEROS:
        raise ValueError(f"--pool debe estar entre {TOP_K} y {TOTAL_NUMEROS}")
    if args.juegos < 1:
        raise ValueError("--juegos debe ser >= 1")
    if args.max_comunes < 0 or args.max_comunes >= TOP_K:
        raise ValueError("--max-comunes debe estar entre 0 y 5")

    print("=" * 72)
    print("TinkaAI Predictor - Generador de combinaciones Ensemble")
    print("=" * 72)

    ranking = cargar_ranking()
    historicas = cargar_historicas()

    pool_df = ranking.head(args.pool)
    pool = sorted(pool_df["Numero"].astype(int).tolist())
    score_por_numero = dict(zip(ranking["Numero"].astype(int), ranking["ScoreFinal"].astype(float)))

    candidatas = []
    descartadas_historicas = 0
    descartadas_estructura = 0

    for combo in itertools.combinations(pool, TOP_K):
        if combo in historicas:
            descartadas_historicas += 1
            continue
        if not es_valida(combo):
            descartadas_estructura += 1
            continue

        promedio, minimo = score_combo(combo, score_por_numero)
        candidatas.append((combo, promedio, minimo))

    candidatas.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    seleccionadas = seleccionar_diversas(candidatas, args.juegos, args.max_comunes)

    if not seleccionadas:
        raise RuntimeError("No se encontraron combinaciones con los filtros actuales")

    filas = []
    for j, (combo, promedio, minimo) in enumerate(seleccionadas, start=1):
        e = estadisticas_combo(combo)
        fila = {
            "Juego": j,
            **{f"Numero{i}": n for i, n in enumerate(combo, start=1)},
            "ScorePromedio": promedio,
            "ScoreMinimo": minimo,
            **e,
        }
        filas.append(fila)

    out = pd.DataFrame(filas)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Pool utilizado: Top {args.pool}")
    print(f"Combinaciones candidatas validas: {len(candidatas)}")
    print(f"Descartadas por repetir historico: {descartadas_historicas}")
    print(f"Descartadas por filtros estructurales: {descartadas_estructura}")
    print(f"Combinaciones generadas: {len(out)}")
    print(f"Archivo: {OUTPUT_FILE}")
    print("-" * 72)

    for _, r in out.iterrows():
        nums = "-".join(f"{int(r[f'Numero{i}']):02d}" for i in range(1, 7))
        print(f"Juego {int(r['Juego']):02d}: {nums} | score={r['ScorePromedio']:.4f}")

    print(
        "Nota: los filtros estructurales sirven para diversificar jugadas; "
        "no convierten el ranking en una garantia de acierto."
    )


if __name__ == "__main__":
    main()
