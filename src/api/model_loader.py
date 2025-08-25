import joblib
import os
from src.pipeline import DataPipeline

class ModelService:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = joblib.load(model_path)
        self.pipeline = DataPipeline() 

    def predict(self, features):
        """Aplica preprocesamiento antes de predecir"""
        processed = self.pipeline.preprocess(features)
        predictions = self.model.predict(processed)
        return predictions.tolist()
