"""Generacion de variables temporales para modelos IA Tinka."""

import pandas as pd


def build_features(input_file, output_file):
    df = pd.read_csv(input_file)

    df["Numero"] = df["Numero"].astype(int)

    if "IdSorteo" in df.columns:
        df = df.sort_values(["Numero", "IdSorteo"]).copy()
    else:
        df = df.sort_values(["Numero"]).copy()

    # Frecuencia acumulada del numero hasta ese momento
    df["FrecuenciaHistorica"] = (
        df.groupby("Numero").cumcount() + 1
    )

    # Ultimo indice temporal conocido por numero
    df["IndiceTemporal"] = df.groupby("Numero").cumcount()

    # Frecuencia movil aproximada sobre apariciones anteriores
    df["FrecuenciaUltimos30"] = (
        df.groupby("Numero")["Aparecio"]
        .rolling(30, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["FrecuenciaUltimos90"] = (
        df.groupby("Numero")["Aparecio"]
        .rolling(90, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df.to_csv(output_file, index=False)

    return df
