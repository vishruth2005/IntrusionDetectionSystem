from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
import pandas as pd
import io
import logging
from src.IDS.training.predict import predict_new_data
import os
import config

ids_router = APIRouter()

@ids_router.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        predictions = predict_new_data(new_df,  config.MODEL_SAVE_PATH, config.PREPROCESSOR_SAVE_PATH, config.MAPPING_SAVE_PATH, config.DEVICE)
        
        output_df = pd.DataFrame({"predictions": predictions})
        output = io.StringIO()
        output_df.to_csv(output, index=False)
        output.seek(0)
        
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))