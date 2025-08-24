# src/pipeline.py
import pandas as pd

class DataPipeline:
    def __init__(self):
        # Aquí podrías cargar encoders, scalers, etc.
        pass

    def preprocess(self, data):
        """
        Recibe datos (lista de listas o DataFrame)
        y devuelve datos preprocesados listos para el modelo.
        """
        # Si es lista de listas, conviértelo en DataFrame temporal
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Ejemplo de pasos de preprocesamiento
        df = self._fill_missing(df)
        df = self._normalize(df)

        return df.values.tolist()  # Devuelve en formato lista para el modelo

    def _fill_missing(self, df):
        """Rellena valores faltantes con 0"""
        return df.fillna(0)

    def _normalize(self, df):
        """Ejemplo simple: divide todo entre 10"""
        return df / 10
