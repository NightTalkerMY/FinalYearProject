import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from allosaurus.app import read_recognizer
import soundfile as sf

app = FastAPI(title="Allosaurus LipSync Microservice")

print("[INIT] Loading Allosaurus Model into RAM...")
recognizer = read_recognizer()
print("[INIT] Ready on Port 8004.")

class AudioRequest(BaseModel):
    audio_path: str

def ipa_to_viseme(ipa_char: str) -> str:
    mapping = {
        'A': ['m', 'b', 'p'], 'B': ['d', 't', 'n', 'k', 'g', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'ŋ', 'h', 'ɹ', 'r', 'ɾ'],
        'C': ['i', 'ɪ', 'e', 'ɛ', 'j'], 'D': ['a', 'ɑ', 'æ', 'ʌ', 'ə', 'ɚ', 'ɝ'],
        'E': ['o', 'ɔ'], 'F': ['u', 'ʊ', 'w'], 'G': ['f', 'v'], 'H': ['l', 'ɫ']
    }
    for viseme, ipa_list in mapping.items():
        if ipa_char in ipa_list: return viseme
    return 'B' 

@app.post("/generate_visemes")
async def generate_visemes(request: AudioRequest):
    audio_path = request.audio_path
    duration = sf.info(audio_path).duration
    
    # Fast 16-bit conversion
    temp_wav = f"temp_{os.path.basename(audio_path)}"
    data, samplerate = sf.read(audio_path)
    sf.write(temp_wav, data, samplerate, subtype='PCM_16')
    
    try:
        raw_output = recognizer.recognize(temp_wav, timestamp=True)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
    mouth_cues = []
    current_time = 0.0
    
    for line in raw_output.splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) != 3: continue
            
        start, phoneme_dur, ipa = float(parts[0]), float(parts[1]), parts[2]
        end = start + phoneme_dur
        
        if start > current_time + 0.01: 
            mouth_cues.append({"start": round(current_time, 2), "end": round(start, 2), "value": "X"})
            
        viseme = ipa_to_viseme(ipa)
        if round(start, 2) < round(end, 2): 
            mouth_cues.append({"start": round(start, 2), "end": round(end, 2), "value": viseme})
            
        current_time = end

    if current_time < duration:
        mouth_cues.append({"start": round(current_time, 2), "end": round(duration, 2), "value": "X"})

    # Return the raw dictionary directly (No saving to disk needed!)
    return {
        "metadata": {
            "soundFile": audio_path,
            "duration": round(duration, 2)
        },
        "mouthCues": mouth_cues
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)