from fastapi import FastAPI, Request
import numpy as np
from faster_whisper import WhisperModel
import uvicorn

print("[SERVER] Starting up...")
app = FastAPI()


print("[SERVER] Loading Whisper model (large-v3)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("[SERVER] Whisper model ready.")

@app.post("/transcribe")
async def transcribe(request: Request):
    body = await request.body()
    audio_int16 = np.frombuffer(body, dtype=np.int16)
    audio_fp32 = audio_int16.astype(np.float32) / 32767.0
    
    # 1. STT Phase
    segments, _ = model.transcribe(audio_fp32, language="en")
    text = "".join(seg.text for seg in segments).strip()
    
    print(f"User Request: {text}")

    # 2. Return the "OK" handshake along with signals
    return {
        "status": "ok", 
        "text": text,
        "trigger_enter": False, # Will be set by LLM logic later
        "viseme": []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)