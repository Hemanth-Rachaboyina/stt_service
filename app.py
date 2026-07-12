import gradio as gr
from faster_whisper import WhisperModel
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SIZE = "tiny"
model = None

# Load model immediately on startup
logger.info(f"Loading Faster Whisper model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
# Warm-up with 1 second of silence (16kHz sample rate)
logger.info("Warming up model...")
dummy_audio = np.zeros(16000, dtype=np.float32)
segments, _ = model.transcribe(dummy_audio, beam_size=1)
logger.info("Model is ready!")


def transcribe_audio(audio):
    try:
        if audio is None:
            return "Please upload an audio file first!"
        logger.info("Starting transcription...")
        segments, _ = model.transcribe(audio, beam_size=5)
        transcription = " ".join([seg.text.strip() for seg in segments])
        logger.info(f"Transcription done: {transcription[:100]}...")
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs=gr.Textbox(label="Transcription", lines=5),
    title="Self-Hosted STT Service",
    description="Powered by Faster Whisper (small model)",
    api_name="stt"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
