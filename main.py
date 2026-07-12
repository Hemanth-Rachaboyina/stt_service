from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STT Service", version="1.0.0")

MODEL_SIZE = "small"
model = None


@app.on_event("startup")
async def startup_event():
    global model
    logger.info(f"Loading Faster Whisper model: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info("Model loaded successfully! Performing warm-up inference...")
    import numpy as np
    dummy_audio = np.zeros(16000, dtype=np.float32)
    model.transcribe(dummy_audio, beam_size=1)
    logger.info("Warm-up complete!")


@app.post("/stt")
async def transcribe_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)
        logger.info(f"Transcribing audio file: {file.filename}")
        segments, _ = model.transcribe(audio_stream, beam_size=5)
        transcription = " ".join([seg.text.strip() for seg in segments])
        logger.info(f"Transcription complete: {transcription[:100]}...")
        return {"text": transcription}
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_SIZE}
