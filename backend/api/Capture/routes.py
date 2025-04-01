from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
import pandas as pd
from src.Capture.processpcap import process_pcap, save_to_arff
from src.Capture.capture import capture_packets_tshark, capture_packets_tshark_wrapper
import config
import io
import logging
import asyncio

class CaptureInput(BaseModel):
    duration: int

capture_router = APIRouter()

@capture_router.post("/", response_class=Response)
async def capture_pcap_file(data: CaptureInput):
    try:
        await asyncio.to_thread(capture_packets_tshark_wrapper, data.duration)
        features = process_pcap(config.PCAP_SAVE_PATH)
        save_to_arff(features, config.PCAP_OUTPUT_PATH)
        df = pd.DataFrame(features)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=Captured_data.csv"})
    except Exception as e:
        logging.error(f"Error capturing pcap file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))