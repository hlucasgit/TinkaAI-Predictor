"""
TinkaAI Predictor - Pipeline principal

Procesa el histórico de Tinka:
1. Carga Excel histórico 20 años
2. Valida números 1-53
3. Normaliza sorteos
4. Genera estadísticas base
5. Genera features para IA

Uso:
    python src/run_pipeline.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.loaders.excel_loader import cargar_excel_tinka
from src.features.build_features import build_features

DATA_RAW = ROOT / "data" / "raw" / "tinka_dataset_historico_20_anios.xlsx"
DATA_PROCESSED = ROOT / "data" / "processed"

MIN_NUMERO = 1
MAX_NUMERO = 53


def main():
    print("=" * 60)
    print("TinkaAI Predictor - Pipeline Histórico 20 años")
    print("=" * 60)

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"No existe: {DATA_RAW}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df_largo = cargar_excel_tinka(DATA_RAW)

    print(f"Registros normalizados: {len(df_largo)}")

    df_largo["Numero"] = df_largo["Numero"].astype(int)

    invalidos = df_largo[
        (df_largo["Numero"] < MIN_NUMERO) |
        (df_largo["Numero"] > MAX_NUMERO)
    ]

    if not invalidos.empty:
        raise ValueError("Se encontraron números fuera del rango permitido 1-53")

    frecuencia = (
        df_largo.groupby("Numero")
        .size()
        .reset_index(name="Apariciones")
        .sort_values("Apariciones", ascending=False)
    )

    sorteos_file = DATA_PROCESSED / "sorteos_largo.csv"
    features_file = DATA_PROCESSED / "features_modelo.csv"

    df_largo.to_csv(sorteos_file, index=False)

    frecuencia.to_csv(
        DATA_PROCESSED / "frecuencia_numeros.csv",
        index=False
    )

    build_features(sorteos_file, features_file)

    print("Proceso terminado correctamente")
    print(f"Archivos generados en: {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
