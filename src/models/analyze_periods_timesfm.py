"""
Analisis por periodos del backtest walk-forward TimesFM.

Divide el resultado de backtesting en bloques para evaluar estabilidad
por periodo temporal.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "backtest_walkforward_timesfm_detalle.csv"
OUTPUT = ROOT / "data" / "processed" / "analisis_periodos_timesfm.csv"
RESUMEN = ROOT / "data" / "processed" / "analisis_periodos_timesfm_resumen.csv"

AZAR_TOP6 = 6 * 6 / 53


def main():
    print("=" * 70)
    print("TinkaAI Predictor - Analisis por periodos TimesFM")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(f"No existe: {INPUT}")

    df = pd.read_csv(INPUT)

    if len(df) == 0:
        raise ValueError("El archivo de backtest esta vacio")

    bloques = []
    tamano = 64

    for inicio in range(0, len(df), tamano):
        bloque = df.iloc[inicio:inicio + tamano].copy()
        if bloque.empty:
            continue

        bloques.append({
            "Bloque": len(bloques) + 1,
            "FechaInicio": bloque["FechaReal"].iloc[0],
            "FechaFin": bloque["FechaReal"].iloc[-1],
            "Sorteos": len(bloque),
            "AciertosTimesFM": int(bloque["AciertosTimesFM"].sum()),
            "PromedioTimesFM": float(bloque["AciertosTimesFM"].mean()),
            "AciertosFrecuencia": int(bloque["AciertosFrecuencia100"].sum()),
            "PromedioFrecuencia": float(bloque["AciertosFrecuencia100"].mean()),
            "AzarEsperadoTop6": AZAR_TOP6,
            "DeltaTimesFMVsAzar": float(bloque["AciertosTimesFM"].mean()) - AZAR_TOP6,
            "DeltaFrecuenciaVsAzar": float(bloque["AciertosFrecuencia100"].mean()) - AZAR_TOP6,
        })

    resultado = pd.DataFrame(bloques)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(OUTPUT, index=False)

    resumen = pd.DataFrame([{
        "BloquesAnalizados": len(resultado),
        "SorteosAnalizados": len(df),
        "PromedioGeneralTimesFM": float(df["AciertosTimesFM"].mean()),
        "PromedioGeneralFrecuencia": float(df["AciertosFrecuencia100"].mean()),
        "AzarEsperadoTop6": AZAR_TOP6,
        "BloquesTimesFMSobreAzar": int((resultado["DeltaTimesFMVsAzar"] > 0).sum()),
        "BloquesFrecuenciaSobreAzar": int((resultado["DeltaFrecuenciaVsAzar"] > 0).sum()),
    }])
    resumen.to_csv(RESUMEN, index=False)

    print(f"Detalle: {OUTPUT}")
    print(f"Resumen: {RESUMEN}")
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
