from fastapi import APIRouter, Body, HTTPException, Response
from pydantic import BaseModel
import pandas as pd
import io
from src.CTGAN.training.generate import generate_samples

ctgan_router = APIRouter()

class GenerateInput(BaseModel):
    num_samples: int
    batch_size: int = 50

@ctgan_router.post("/generate", response_class=Response)
async def generate(data: GenerateInput):
    try:
        generated_data = generate_samples(data.num_samples, data.batch_size)
        df = pd.DataFrame(generated_data)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
