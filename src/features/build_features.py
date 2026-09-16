"""Generacion de variables para modelos IA Tinka."""

from pathlib import Path
import pandas as pd


def build_features(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Numero'] = df['Numero'].astype(int)

    df = df.sort_values(['Numero']).copy()
    df['FrecuenciaHistorica'] = df.groupby('Numero').cumcount() + 1

    ultima = df.groupby('Numero').cumcount()
    df['IndiceTemporal'] = ultima

    df.to_csv(output_file, index=False)
    return df
