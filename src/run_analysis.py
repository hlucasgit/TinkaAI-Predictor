"""
Punto de entrada del analisis TinkaAI.

Ejecuta el flujo:
1. Carga historico Excel
2. Limpieza de datos
3. Transformacion analitica
4. Generacion de resultados
"""

from pathlib import Path

from ingestion.excel_loader import cargar_excel
from analytics.frequency import transformar_a_formato_largo, frecuencia_numeros


RAW_FILE = Path("data/raw/tinka_dataset.xlsx")
OUTPUT_DIR = Path("data/processed")


def ejecutar():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_excel(RAW_FILE)

    df_largo = transformar_a_formato_largo(df)
    df_frecuencia = frecuencia_numeros(df)

    df_largo.to_csv(
        OUTPUT_DIR / "sorteos_largo.csv",
        index=False
    )

    df_frecuencia.to_csv(
        OUTPUT_DIR / "frecuencia_numeros.csv",
        index=False
    )

    print("Analisis generado correctamente")


if __name__ == "__main__":
    ejecutar()
