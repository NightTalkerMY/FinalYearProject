import os
import subprocess
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path

# --- CRITICAL FFmpeg FIX FOR TORCHCODEC ---
# Point this exactly to your ffmpeg bin folder
FFMPEG_BIN_PATH = r"C:\Users\wwon0076\Desktop\FYP\ffmpeg\bin"

if os.path.exists(FFMPEG_BIN_PATH):
    # This is the modern way to add DLLs to Python 3.8+ on Windows
    os.add_dll_directory(FFMPEG_BIN_PATH)
    # Also update PATH for subprocesses like Rhubarb
    os.environ["PATH"] = FFMPEG_BIN_PATH + os.pathsep + os.environ["PATH"]
else:
    print(f"CRITICAL ERROR: FFmpeg not found at {FFMPEG_BIN_PATH}")
# ------------------------------------------

# Import your existing class logic
# (Assuming your provided code is in a file named xtts_logic.py in the same folder)
from TTS_fyp import XTTSEngine 

app = FastAPI(title="PUMA Holographic Assistant - TTS Service")

# --- CONFIGURATION ---
RHUBARB_PATH = str(Path("../rhubarb/rhubarb.exe").resolve())
OUTPUT_DIR = Path("outputs_xtts")
SPEAKER_WAV = "sample.wav" ## clone voice, None use default

# Initialize the engine once on startup
print("[TTS] Loading XTTS Model to GPU...")
engine = XTTSEngine(
    speaker_wav=SPEAKER_WAV,
    speaker="Ana Florence",
    voice_mode="auto"
)

@app.post("/generate_speech")
async def generate_speech(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    if not text:
        return {"error": "No text provided"}

    # 1. Generate the Audio File (.wav)
    # This uses your existing logic to save a timestamped file
    wav_file_path = engine.speak(text)
    
    # 2. Run Rhubarb for Lip-Sync (.json)
    # We name the json the same as the wav file
    json_file_path = wav_file_path.replace(".wav", ".json")
    
    print(f"[RHUBARB] Generating visemes for {Path(wav_file_path).name}...")
    try:
        # Command: rhubarb.exe -f json -o output.json input.wav
        subprocess.run([
            RHUBARB_PATH, 
            "-f", "json", 
            "-o", json_file_path, 
            wav_file_path
        ], check=True)
    except Exception as e:
        print(f"[ERROR] Rhubarb failed: {e}")

    # 3. Return paths to the Orchestrator
    # The Orchestrator will then tell React where to find these files
    return {
        "status": "success",
        "audio_path": str(Path(wav_file_path).absolute()),
        "viseme_path": str(Path(json_file_path).absolute()),
        "filename": Path(wav_file_path).name
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)