"""
TinkaAI Predictor - Pipeline principal

Ejecuta el flujo completo:
1. Carga histórico Tinka
2. Valida datos
3. Genera dataset analítico
4. Calcula estadísticas
5. Genera features para IA

Uso:
    python src/run_pipeline.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


DATA_RAW = ROOT / "data" / "raw" / "tinka_dataset.xlsx"
DATA_PROCESSED = ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("TinkaAI Predictor - Pipeline")
    print("=" * 60)

    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo histórico: {DATA_RAW}"
        )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Archivo origen: {DATA_RAW}")
    print(f"Salida: {DATA_PROCESSED}")

    print("\nPipeline preparado correctamente.")
    print("Siguiente etapa: conectar módulos de carga, análisis y modelos.")


if __name__ == "__main__":
    main()
