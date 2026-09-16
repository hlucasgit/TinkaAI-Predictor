"""
Integracion inicial con TimesFM.

Este modulo prepara la capa de adaptacion para modelos de series temporales.
TimesFM sera utilizado sobre series temporales generadas desde datos historicos.
"""

from typing import List

import pandas as pd


class TimesFMModel:
    """
    Wrapper inicial para TimesFM.

    La implementacion real del modelo se activara cuando el entorno
    tenga instalada la version compatible de timesfm y sus dependencias.
    """

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.model = None

    def preparar_serie(
        self,
        df: pd.DataFrame,
        fecha: str = "Fecha_sorteo",
        valor: str = "Aparecio",
    ) -> pd.DataFrame:
        """
        Prepara una serie temporal compatible con modelos forecast.
        """

        serie = df[[fecha, valor]].copy()
        serie[fecha] = pd.to_datetime(serie[fecha])
        serie = serie.sort_values(fecha)

        return serie

    def cargar_modelo(self):
        """
        Punto de extension para carga del modelo TimesFM.
        """

        try:
            import timesfm
            self.model = timesfm
            return True
        except ImportError:
            return False

    def predecir(self, valores: List[float]):
        """
        Ejecutara el forecast cuando el modelo TimesFM este cargado.
        """

        if self.model is None:
            raise RuntimeError("TimesFM no esta cargado")

        return {
            "horizon": self.horizon,
            "input_length": len(valores),
            "status": "modelo preparado"
        }
