import time
import queue
import numpy as np
import sounddevice as sd
import requests
from pathlib import Path
from collections import deque
from openwakeword.model import Model

# --- CONFIGURATION ---
DESKTOP_IP = "100.100.155.39"  
URL = f"http://{DESKTOP_IP}:5000/process"

# Audio Engine Config
NATIVE_RATE = 48000
TARGET_RATE = 16000
CHUNK_SAMPLES = 1280  
RESAMPLE_STEP = NATIVE_RATE // TARGET_RATE
BUFFER_SIZE = CHUNK_SAMPLES * RESAMPLE_STEP

# Wake Word Config
WAKE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0 

# Speech Endpointing
SILENCE_END_SEC = 1.0
MAX_LISTEN_SEC = 10.0      # Safety timeout
PREROLL_SEC = 0.5
PREROLL_FRAMES = int(PREROLL_SEC / (CHUNK_SAMPLES / TARGET_RATE))

# Global threshold variable (set by calibration)
RMS_SPEECH_THRESHOLD = 0.015 

def rms(x_float32: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x_float32)) + 1e-12))

def get_usb_mic_index():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
            return i
    return None

def calibrate_noise_floor(audio_q, n_frames=20):
    print("Calibrating background noise... (shhh!)")
    levels = []
    while not audio_q.empty(): audio_q.get()
    
    for _ in range(n_frames):
        raw = audio_q.get()
        frame_16k = raw[::RESAMPLE_STEP][:CHUNK_SAMPLES]
        levels.append(rms(frame_16k))
    
    avg_noise = np.mean(levels)
    new_thresh = avg_noise + 0.02 # Add margin
    print(f"Calibration Complete. Noise: {avg_noise:.4f} -> Threshold: {new_thresh:.4f}")
    return new_thresh

def main():
    print("Loading Puma Wake Word...")
    model_path = str(Path(__file__).parent / "hey_holo.onnx")
    try:
        oww = Model(wakeword_models=[model_path])
    except TypeError:
        oww = Model([model_path])

    target_id = get_usb_mic_index()
    if target_id is None:
        print("CRITICAL: USB Mic not found.")
        return

    print(f"Connected to Mic {target_id} at {NATIVE_RATE}Hz")

    audio_q = queue.Queue()
    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    cooldown_frames = int((COOLDOWN_SECONDS * NATIVE_RATE) / BUFFER_SIZE)
    current_cooldown = 0
    preroll = deque(maxlen=PREROLL_FRAMES)
    
    state = "IDLE"
    utter_frames = []
    last_speech_time = 0
    listen_start_time = 0 
    
    global RMS_SPEECH_THRESHOLD 

    print("System Ready. Waiting for 'Hey Puma'...")
    
    with sd.InputStream(device=target_id, samplerate=NATIVE_RATE, channels=1, 
                        blocksize=BUFFER_SIZE, callback=callback):
        
        RMS_SPEECH_THRESHOLD = calibrate_noise_floor(audio_q)

        while True:
            raw_frame = audio_q.get()
            frame_16k = raw_frame[::RESAMPLE_STEP][:CHUNK_SAMPLES]
            preroll.append(frame_16k)
            frame_i16 = (np.clip(frame_16k, -1.0, 1.0) * 32767.0).astype(np.int16)

            if state == "IDLE":
                prediction = oww.predict(frame_i16)
                
                if current_cooldown > 0:
                    current_cooldown -= 1
                    continue

                score = list(prediction.values())[0]
                if score >= WAKE_THRESHOLD:
                    print(f"\n[WAKE] Detected! (Score: {score:.2f})")
                    state = "LISTENING"
                    if hasattr(oww, 'reset'): oww.reset()
                    
                    # [TRIM FIX] Start empty to remove "Hey Puma"
                    utter_frames = [] 
                    
                    last_speech_time = time.time()
                    listen_start_time = time.time()

            elif state == "LISTENING":
                utter_frames.append(frame_16k)
                
                current_rms = rms(frame_16k)
                print(f"RMS: {current_rms:.4f} | Limit: {RMS_SPEECH_THRESHOLD:.4f}", end='\r')

                if current_rms >= RMS_SPEECH_THRESHOLD:
                    last_speech_time = time.time()
                
                is_silence_timeout = (time.time() - last_speech_time) >= SILENCE_END_SEC
                is_hard_timeout = (time.time() - listen_start_time) >= MAX_LISTEN_SEC

                if is_silence_timeout or is_hard_timeout:
                    print(f"\n[END] Processing...")
                    state = "PROCESSING"

            elif state == "PROCESSING":
                utter = np.concatenate(utter_frames).astype(np.float32)
                audio_payload = (np.clip(utter, -1.0, 1.0) * 32767).astype(np.int16)
                
                try:
                    r = requests.post(URL, data=audio_payload.tobytes(), timeout=20)
                    response = r.json()
                    if response.get("status") == "ok":
                        print(f"[DESKTOP] {response.get('text')}")
                        if response.get("trigger_enter"):
                            print(">>> TRIGGER: ENTER CAROUSEL MODE <<<")
                except Exception as e:
                    print(f"[ERROR] Connection failed: {e}")

                while not audio_q.empty(): audio_q.get_nowait()
                if hasattr(oww, 'reset'): oww.reset()
                current_cooldown = cooldown_frames
                state = "IDLE"
                utter_frames = []
                print("Ready.")

if __name__ == "__main__":
    main()
