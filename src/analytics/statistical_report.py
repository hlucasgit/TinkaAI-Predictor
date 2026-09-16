"""Reportes estadisticos del historico Tinka."""

import pandas as pd


def generar_reporte(input_file, output_file):
    df = pd.read_csv(input_file)

    reporte = []

    frecuencia = (
        df.groupby("Numero")
        .size()
        .reset_index(name="Apariciones")
        .sort_values("Apariciones", ascending=False)
    )

    for _, row in frecuencia.iterrows():
        reporte.append({
            "Tipo": "Frecuencia",
            "Numero": int(row["Numero"]),
            "Valor": int(row["Apariciones"])
        })

    resultado = pd.DataFrame(reporte)
    resultado.to_csv(output_file, index=False)

    return resultado
