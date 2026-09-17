"""Convierte el forecast TimesFM en rankings de numeros candidatos.

IMPORTANTE: los valores de TimesFM se tratan como scores de forecast,
no como probabilidades calibradas de ganar.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "prediccion_timesfm.csv"
RANKING_FILE = ROOT / "data" / "processed" / "ranking_timesfm.csv"
COMBINACIONES_FILE = ROOT / "data" / "processed" / "combinaciones_timesfm.csv"

TOP_K = 6


def _numero_desde_columna(columna: str) -> int:
    return int(columna[1:])


def generar_rankings(df: pd.DataFrame):
    columnas = [c for c in df.columns if c.startswith("N")]
    if len(columnas) != 53:
        raise ValueError(f"Se esperaban 53 series N01..N53 y se encontraron {len(columnas)}")

    filas_ranking = []
    filas_combinaciones = []

    for idx, fila in df.iterrows():
        paso = int(fila["PasoForecast"]) if "PasoForecast" in df.columns else idx + 1
        pares = sorted(
            [(_numero_desde_columna(c), float(fila[c])) for c in columnas],
            key=lambda x: x[1],
            reverse=True,
        )

        for posicion, (numero, score) in enumerate(pares, start=1):
            filas_ranking.append({
                "PasoForecast": paso,
                "Posicion": posicion,
                "Numero": numero,
                "ScoreTimesFM": score,
            })

        top = pares[:TOP_K]
        numeros = sorted(numero for numero, _ in top)
        fila_comb = {"PasoForecast": paso}
        for i, numero in enumerate(numeros, start=1):
            fila_comb[f"Numero{i}"] = numero
        fila_comb["ScorePromedioTop6"] = sum(score for _, score in top) / TOP_K
        filas_combinaciones.append(fila_comb)

    return pd.DataFrame(filas_ranking), pd.DataFrame(filas_combinaciones)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)
    ranking, combinaciones = generar_rankings(df)

    ranking.to_csv(RANKING_FILE, index=False)
    combinaciones.to_csv(COMBINACIONES_FILE, index=False)

    print(f"Ranking generado: {RANKING_FILE}")
    print(f"Combinaciones candidatas: {COMBINACIONES_FILE}")
    print("Nota: ScoreTimesFM es un score de forecast, no una probabilidad calibrada.")


if __name__ == "__main__":
    main()
