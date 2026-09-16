"""
Capa de evaluación para modelos TimesFM.

Prepara series temporales y permite comparar resultados
contra valores históricos.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimesFMForecaster:
    """
    Adaptador inicial para trabajar con TimesFM.

    La integración directa con el modelo se mantiene desacoplada
    para permitir cambiar versiones del framework.
    """

    def preparar_serie(self, df: pd.DataFrame, numero: int) -> pd.DataFrame:
        """Genera serie temporal de aparición para un número."""

        serie = df[df["Numero"] == numero].copy()
        serie = serie.sort_values("Fecha_sorteo")

        return serie[["Fecha_sorteo", "Aparecio"]]

    def dividir_train_test(self, serie: pd.DataFrame, porcentaje: float = 0.8):
        """Divide cronológicamente la serie."""

        limite = int(len(serie) * porcentaje)

        return (
            serie.iloc[:limite],
            serie.iloc[limite:]
        )

    def evaluar(self, real, prediccion):
        """Calcula métricas básicas."""

        return {
            "MAE": mean_absolute_error(real, prediccion),
            "RMSE": mean_squared_error(real, prediccion) ** 0.5,
        }
