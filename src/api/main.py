from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Any
from src.model_loader import ModelService
import pandas as pd
import io

app = FastAPI(title="Simple Model API")

# Inicializar modelo al arrancar el servidor
model_service = ModelService("src/models/my_model.pkl")

# "Base de datos" temporal en memoria
database = []

# Schema para validación de request
class PredictRequest(BaseModel):
    features: List[List[float]]  # ejemplo: [[1.0, 2.0, 3.0]]

class PredictResponse(BaseModel):
    predictions: List[Any]

@app.get("/")
def root():
    return {"message": "API funcionando"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        preds = model_service.predict(request.features)
        # Guardar datos en memoria (simulando backend)
        database.append({"input": request.features, "prediction": preds})
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Recibe un CSV, predice y devuelve un CSV con resultados"""
    try:
        # Leer CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validar que hay columnas
        if df.empty:
            raise HTTPException(status_code=400, detail="El CSV está vacío")

        # Extraer features
        features = df.values.tolist()
        preds = model_service.predict(features)

        # Agregar predicciones
        df["prediction"] = preds

        # Guardar en memoria
        database.append({"file": file.filename, "rows": len(df)})

        # Convertir a CSV en memoria
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/records")
def get_records():
    """Devuelve todos los registros guardados"""
    return database

