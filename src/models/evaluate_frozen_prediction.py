"""Evalua una prediccion prospectiva congelada contra un sorteo real.

Por defecto lee:
data/processed/prediccion_congelada_actual.json

Uso de prueba (NO guarda):
  python src/models/evaluate_frozen_prediction.py --numbers 1 2 3 4 5 6

Uso con resultado real manual (requiere confirmacion explicita):
  python src/models/evaluate_frozen_prediction.py --numbers 1 2 3 4 5 6 --result-date 2026-09-20 --save --confirm-real

Tambien puede evaluar una fecha ya registrada en:
data/raw/tinka_actualizacion_2026.csv

  python src/models/evaluate_frozen_prediction.py --date 2026-09-20

No modifica el snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "data" / "processed" / "prediccion_congelada_actual.json"
UPDATE_FILE = ROOT / "data" / "raw" / "tinka_actualizacion_2026.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "evaluaciones_prospectivas.csv"

TOP_K = 6


def cargar_snapshot(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validar_numeros(nums: list[int]) -> list[int]:
    nums = sorted(int(x) for x in nums)
    if len(nums) != TOP_K:
        raise ValueError("Se requieren exactamente 6 numeros")
    if len(set(nums)) != TOP_K:
        raise ValueError("Los 6 numeros deben ser distintos")
    if min(nums) < 1 or max(nums) > 53:
        raise ValueError("Los numeros deben estar entre 1 y 53")
    return nums


def cargar_por_fecha(fecha: str) -> list[int]:
    if not UPDATE_FILE.exists():
        raise FileNotFoundError(UPDATE_FILE)

    df = pd.read_csv(UPDATE_FILE)
    if "Fecha" not in df.columns:
        raise ValueError("El CSV de actualizaciones no contiene Fecha")

    fila = df.loc[df["Fecha"].astype(str) == str(fecha)]
    if fila.empty:
        raise ValueError(f"No se encontro la fecha {fecha} en {UPDATE_FILE}")
    if len(fila) > 1:
        raise ValueError(f"Hay mas de un registro para la fecha {fecha}")

    r = fila.iloc[0]
    return validar_numeros([int(r[f"Numero{i}"]) for i in range(1, 7)])


def hits(pred: list[int], reales: list[int]) -> tuple[int, list[int]]:
    ac = sorted(set(pred) & set(reales))
    return len(ac), ac


def evaluar(snapshot: dict, reales: list[int]) -> dict:
    top6 = [int(x) for x in snapshot.get("top6_determinista", [])]
    if len(top6) != TOP_K:
        raise ValueError("El snapshot no contiene top6_determinista valido")

    corte = snapshot.get("analisis_corte_top6", {})
    core = [int(x) for x in corte.get("numeros_sobre_corte", [])]
    empatados = [int(x) for x in corte.get("numeros_empatados_en_corte", [])]
    plazas = int(corte.get("plazas_restantes_dentro_empate", 0))

    h_top6, ac_top6 = hits(top6, reales)

    h_core, ac_core = hits(core, reales)
    h_emp, ac_emp = hits(empatados, reales)

    esperado_tie = float(h_core)
    if empatados and plazas > 0:
        esperado_tie += plazas * (h_emp / len(empatados))

    tfm = [int(x) for x in snapshot.get("top6_timesfm_diagnostico", [])]
    h_tfm, ac_tfm = hits(tfm, reales) if len(tfm) == TOP_K else (None, [])

    freq = [int(x) for x in snapshot.get("top6_frecuencia_reciente", [])]
    h_freq, ac_freq = hits(freq, reales) if len(freq) == TOP_K else (None, [])

    combos = []
    for item in snapshot.get("combinaciones", []):
        nums = [int(item[f"Numero{i}"]) for i in range(1, 7)]
        h, ac = hits(nums, reales)
        combos.append(
            {
                "Juego": int(item.get("Juego", len(combos) + 1)),
                "Numeros": nums,
                "Aciertos": h,
                "Acertados": ac,
            }
        )

    mejor = max(combos, key=lambda x: x["Aciertos"]) if combos else None

    return {
        "top6_determinista": top6,
        "aciertos_top6": h_top6,
        "numeros_acertados_top6": ac_top6,
        "core_sobre_corte": core,
        "aciertos_core": h_core,
        "empatados_corte": empatados,
        "aciertos_dentro_empate": h_emp,
        "esperado_tie_aware": esperado_tie,
        "top6_timesfm_diagnostico": tfm,
        "aciertos_timesfm": h_tfm,
        "numeros_acertados_timesfm": ac_tfm,
        "top6_frecuencia_reciente": freq,
        "aciertos_frecuencia": h_freq,
        "numeros_acertados_frecuencia": ac_freq,
        "mejor_combinacion": mejor,
        "combinaciones": combos,
    }


def append_ledger(snapshot: dict, reales: list[int], resultado: dict, fecha: str | None):
    fila = {
        "FechaEvaluada": fecha or "",
        "UltimaFechaHistoricaSnapshot": snapshot.get("ultima_fecha_historica", ""),
        "NumerosReales": "-".join(f"{n:02d}" for n in reales),
        "Top6Determinista": "-".join(f"{n:02d}" for n in resultado["top6_determinista"]),
        "AciertosTop6": resultado["aciertos_top6"],
        "EsperadoTieAware": resultado["esperado_tie_aware"],
        "Top6Frecuencia": "-".join(
            f"{n:02d}" for n in resultado["top6_frecuencia_reciente"]
        ),
        "AciertosFrecuencia": resultado["aciertos_frecuencia"],
        "Top6TimesFM": "-".join(
            f"{n:02d}" for n in resultado["top6_timesfm_diagnostico"]
        ),
        "AciertosTimesFM": resultado["aciertos_timesfm"],
        "MejorJuego": (
            resultado["mejor_combinacion"]["Juego"]
            if resultado["mejor_combinacion"]
            else None
        ),
        "MejorJuegoAciertos": (
            resultado["mejor_combinacion"]["Aciertos"]
            if resultado["mejor_combinacion"]
            else None
        ),
    }

    nuevo = pd.DataFrame([fila])
    if OUTPUT_FILE.exists():
        anterior = pd.read_csv(OUTPUT_FILE)
        if fecha:
            anterior = anterior.loc[anterior["FechaEvaluada"].astype(str) != str(fecha)]
        out = pd.concat([anterior, nuevo], ignore_index=True)
    else:
        out = nuevo

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Evalua un snapshot prospectivo congelado")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--numbers", nargs=6, type=int)
    g.add_argument("--date", type=str)
    p.add_argument("--snapshot", type=str, default=str(DEFAULT_SNAPSHOT))
    p.add_argument(
        "--result-date",
        type=str,
        default=None,
        help="Fecha real YYYY-MM-DD cuando --numbers corresponde a un sorteo real",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Guarda la evaluacion en el ledger. Sin --save la ejecucion es solo prueba.",
    )
    p.add_argument(
        "--confirm-real",
        action="store_true",
        help="Confirma que --numbers corresponde a un resultado real ya ocurrido.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    snapshot_path = Path(args.snapshot)
    snapshot = cargar_snapshot(snapshot_path)

    if args.numbers is not None:
        reales = validar_numeros(args.numbers)
        fecha = args.result_date
    else:
        reales = cargar_por_fecha(args.date)
        fecha = args.date

    if args.save and not fecha:
        raise ValueError(
            "Para guardar una evaluacion con --numbers debe indicar --result-date YYYY-MM-DD"
        )

    if fecha:
        fecha_obj = pd.to_datetime(fecha, errors="raise").date()
        hoy = date.today()
        fecha_snapshot = pd.to_datetime(
            snapshot.get("ultima_fecha_historica"), errors="raise"
        ).date()

        if fecha_obj <= fecha_snapshot:
            raise ValueError(
                f"La fecha evaluada {fecha_obj} debe ser posterior al snapshot "
                f"({fecha_snapshot})."
            )
        if fecha_obj > hoy:
            raise ValueError(
                f"No se puede guardar/evaluar como real una fecha futura: {fecha_obj}. "
                f"Hoy es {hoy}."
            )

    if args.save and args.numbers is not None and not args.confirm_real:
        raise ValueError(
            "Para guardar numeros manuales debe agregar --confirm-real. "
            "Esto evita registrar ejemplos o pruebas como resultados oficiales."
        )

    r = evaluar(snapshot, reales)

    print("=" * 78)
    print("TinkaAI Predictor - Evaluacion prospectiva congelada")
    print("=" * 78)
    print(f"Snapshot: {snapshot_path}")
    print(f"Ultima fecha historica del snapshot: {snapshot.get('ultima_fecha_historica')}")
    if fecha:
        print(f"Fecha evaluada: {fecha}")
    print(f"Resultado real: {reales}")
    print("-" * 78)
    print(
        f"Top-6 determinista: {r['top6_determinista']} | "
        f"aciertos={r['aciertos_top6']} | "
        f"acertados={r['numeros_acertados_top6']}"
    )
    print(
        f"Frecuencia reciente: {r['top6_frecuencia_reciente']} | "
        f"aciertos={r['aciertos_frecuencia']}"
    )
    if r["top6_timesfm_diagnostico"]:
        print(
            f"TimesFM diagnostico: {r['top6_timesfm_diagnostico']} | "
            f"aciertos={r['aciertos_timesfm']}"
        )
    print(
        "Tie-aware: "
        f"core={r['core_sobre_corte']}, "
        f"empatados={r['empatados_corte']}, "
        f"acierto esperado={r['esperado_tie_aware']:.4f}"
    )

    if r["mejor_combinacion"]:
        m = r["mejor_combinacion"]
        print(
            f"Mejor de las combinaciones congeladas: Juego {m['Juego']} | "
            f"{m['Numeros']} | aciertos={m['Aciertos']} | acertados={m['Acertados']}"
        )

    if args.save:
        append_ledger(snapshot, reales, r, fecha)
        print(f"Ledger actualizado: {OUTPUT_FILE}")
    else:
        print("Modo prueba: no se modifico el ledger. Use --save solo con un resultado real.")


if __name__ == "__main__":
    main()
