from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.IDS.routes import ids_router
from api.CTGAN.routes import ctgan_router
from api.Capture.routes import capture_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello!"}

app.include_router(ids_router, prefix='/predict', tags=['predict'])
app.include_router(ctgan_router, prefix='/generate', tags=['generate'])
app.include_router(capture_router, prefix='/capture', tags=['capture'])