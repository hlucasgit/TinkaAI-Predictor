"""
Analisis de patrones historicos de sorteos.
"""

import pandas as pd


NUMBER_COLUMNS = [
    "Numero_1",
    "Numero_2",
    "Numero_3",
    "Numero_4",
    "Numero_5",
    "Numero_6",
]


def numeros_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula cuantos sorteos han pasado desde la ultima aparicion.
    """

    ultimo = {}
    contador = {numero: 0 for numero in range(1, 49)}

    for _, row in df.sort_values("Fecha_sorteo").iterrows():
        contador = {k: v + 1 for k, v in contador.items()}

        for columna in NUMBER_COLUMNS:
            numero = int(row[columna])
            contador[numero] = 0
            ultimo[numero] = row["Fecha_sorteo"]

    return pd.DataFrame(
        [
            {
                "Numero": numero,
                "Sorteos_Sin_Aparecer": ausencia,
                "Ultima_Aparicion": ultimo.get(numero),
            }
            for numero, ausencia in contador.items()
        ]
    ).sort_values("Sorteos_Sin_Aparecer", ascending=False)
