from fastapi import FastAPI, Request
import numpy as np
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="PUMA Holographic Assistant - STT Service")

print("[SERVER] Loading Whisper model (large-v3)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("[SERVER] Whisper model ready.")

@app.post("/transcribe")
async def transcribe(request: Request):
    # Get raw PCM audio bytes from the Raspberry Pi / Orchestrator
    body = await request.body()
    
    # Convert buffer to the format Whisper expects
    audio_int16 = np.frombuffer(body, dtype=np.int16)
    audio_fp32 = audio_int16.astype(np.float32) / 32767.0
    
    # 2. Transcription Phase
    # We specify language="en" to avoid the model "guessing" and adding latency
    segments, _ = model.transcribe(audio_fp32, language="en", beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    
    print(f"[STT Result] {text}")

    # 3. Simple JSON response back to the Orchestrator
    return {
        "text": text
    }

if __name__ == "__main__":
    # Standardizing port 8000 for STT
    uvicorn.run(app, host="0.0.0.0", port=8000)