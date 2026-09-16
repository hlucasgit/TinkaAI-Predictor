"""
TinkaAI Predictor - Pipeline principal

Procesa el histórico de Tinka:
1. Carga Excel histórico 20 años
2. Valida números 1-53
3. Normaliza sorteos
4. Genera estadísticas y features IA

Uso:
    python src/run_pipeline.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw" / "tinka_dataset_historico_20_anios.xlsx"
DATA_PROCESSED = ROOT / "data" / "processed"

MIN_NUMERO = 1
MAX_NUMERO = 53


def detectar_columnas_numero(df):
    return [c for c in df.columns if 'numero' in c.lower() or 'n' in c.lower()]


def main():
    print('=' * 60)
    print('TinkaAI Predictor - Pipeline Histórico 20 años')
    print('=' * 60)

    if not DATA_RAW.exists():
        raise FileNotFoundError(f'No existe: {DATA_RAW}')

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_RAW)

    print(f'Registros encontrados: {len(df)}')
    print(f'Columnas: {list(df.columns)}')

    columnas_numero = detectar_columnas_numero(df)

    if len(columnas_numero) < 6:
        raise ValueError('No se encontraron las 6 columnas de números del sorteo')

    largo = df.melt(
        value_vars=columnas_numero[:6],
        var_name='Posicion',
        value_name='Numero'
    )

    largo['Numero'] = pd.to_numeric(largo['Numero'], errors='coerce')

    invalidos = largo[(largo['Numero'] < MIN_NUMERO) | (largo['Numero'] > MAX_NUMERO)]

    if len(invalidos) > 0:
        print(f'Advertencia: {len(invalidos)} números fuera del rango 1-53')

    largo = largo.dropna()
    largo = largo[(largo['Numero'] >= MIN_NUMERO) & (largo['Numero'] <= MAX_NUMERO)]

    frecuencia = (
        largo.groupby('Numero')
        .size()
        .reset_index(name='Apariciones')
        .sort_values('Apariciones', ascending=False)
    )

    largo.to_csv(DATA_PROCESSED / 'sorteos_largo.csv', index=False)
    frecuencia.to_csv(DATA_PROCESSED / 'frecuencia_numeros.csv', index=False)

    print('Proceso terminado correctamente')
    print(f'Salidas generadas en: {DATA_PROCESSED}')


if __name__ == '__main__':
    main()
