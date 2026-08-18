# import subprocess
# import uvicorn
# from datetime import datetime
# from fastapi import FastAPI, Request
# from pathlib import Path

# # Import the ultra-fast persistent engine
# from engine import ZipVoiceEngine, SHORT_WAV, PROMPT_TEXT

# app = FastAPI(title="PUMA Holographic Assistant - ZipVoice TTS Service")

# # --- CONFIGURATION ---
# RHUBARB_PATH = str(Path("../../rhubarb/rhubarb.exe").resolve())
# OUTPUT_DIR = Path("outputs_zipvoice")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

# # Initialize the engine ONCE on startup
# print("[TTS] Loading ZipVoice Distill Model to RTX 4080 Super VRAM...")
# tts_engine = ZipVoiceEngine(prompt_wav=SHORT_WAV, prompt_text=PROMPT_TEXT)

# @app.post("/generate_speech")
# async def generate_speech(request: Request):
#     data = await request.json()
#     text = data.get("text", "")
    
#     if not text:
#         return {"error": "No text provided"}

#     # Create a unique timestamped filename
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     wav_file_path = str(OUTPUT_DIR / f"{ts}.wav")
#     json_file_path = str(OUTPUT_DIR / f"{ts}.json")

#     # Generate the Audio File (.wav) instantly via VRAM
#     tts_engine.speak(text, wav_file_path)
    
#     # Run Rhubarb for Lip-Sync (.json)
#     print(f"[RHUBARB] Generating visemes for {Path(wav_file_path).name}...")
#     try:
#         subprocess.run([
#             RHUBARB_PATH, 
#             "-f", "json", 
#             "-o", json_file_path, 
#             wav_file_path
#         ], check=True)
#     except Exception as e:
#         print(f"[ERROR] Rhubarb failed: {e}")

#     return {
#         "status": "success",
#         "audio_path": str(Path(wav_file_path).absolute()),
#         "viseme_path": str(Path(json_file_path).absolute()),
#         "filename": Path(wav_file_path).name
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8003)

import os
import uvicorn
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from pathlib import Path
from engine import ZipVoiceEngine, SHORT_WAV, PROMPT_TEXT

app = FastAPI(title="PUMA Holographic Assistant - ZipVoice TTS Service")

OUTPUT_DIR = Path("outputs_zipvoice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

print("[TTS] Loading ZipVoice Distill Model to RTX 4080 Super VRAM...")
tts_engine = ZipVoiceEngine(prompt_wav=SHORT_WAV, prompt_text=PROMPT_TEXT)

# The URL of your new Allosaurus microservice
ALLOSAURUS_URL = os.getenv(
    "HOLOPI_ALLOSAURUS_URL",
    "http://127.0.0.1:8004/generate_visemes",
)

@app.post("/generate_speech")
async def generate_speech(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    if not text:
        return {"error": "No text provided"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_file_path = str((OUTPUT_DIR / f"{ts}.wav").absolute())
    # --- MISSING LINE ADDED HERE ---
    json_file_path = str((OUTPUT_DIR / f"{ts}.json").absolute()) 

    # 1. Generate Audio via VRAM
    tts_engine.speak(text, wav_file_path)
    
    # 2. Ping the Allosaurus Microservice instantly
    viseme_data = {}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ALLOSAURUS_URL, json={"audio_path": wav_file_path}, timeout=10.0)
            if response.status_code == 200:
                viseme_data = response.json()
                
                # 3. Save the file so React can download it
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    # Added indent=2 so it's readable if you ever need to debug it
                    json.dump(viseme_data, f, indent=2)

        except Exception as e:
            print(f"[ERROR] Could not connect to Allosaurus or save JSON: {e}")

    # 4. Send EVERYTHING back to the frontend
    return {
        "status": "success",
        "audio_path": wav_file_path,
        "filename": Path(wav_file_path).name,
        "visemes": viseme_data  
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
