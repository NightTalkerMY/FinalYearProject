"""
Test script: Record audio from USB mic and save to WAV file.
Run this on the Pi where the mic is connected.
Usage: python test_capture_audio.py [seconds]
       Default: 10 seconds
"""
import sys
import wave
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

CAPTURE_RATE = 48000  # USB mic native rate
TARGET_RATE = 16000   # What openwakeword expects
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUTPUT_FILE = "test_pi_audio.wav"


def get_usb_mic_index():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
            return i, dev['name']
    return None, None


def main():
    mic_idx, mic_name = get_usb_mic_index()
    if mic_idx is None:
        print("[FAIL] No USB mic found! Available devices:")
        print(sd.query_devices())
        return

    print(f"Using: [{mic_idx}] {mic_name}")
    print(f"Recording {DURATION}s at {TARGET_RATE}Hz...")
    print("Speak now!\n")

    audio = sd.rec(
        int(DURATION * CAPTURE_RATE),
        samplerate=CAPTURE_RATE,
        channels=1,
        dtype='float32',
        device=mic_idx
    )
    sd.wait()

    # Downsample 48kHz -> 16kHz
    audio_f32 = audio.flatten()
    audio_16k = resample_poly(audio_f32, 1, CAPTURE_RATE // TARGET_RATE)

    rms = np.sqrt(np.mean(audio_16k ** 2) + 1e-12)
    audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(OUTPUT_FILE, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(audio_int16.tobytes())

    print(f"\n[SAVED] {OUTPUT_FILE} ({DURATION}s, {TARGET_RATE}Hz, mono)")
    print(f"  RMS: {rms:.6f}")
    print(f"  Max: {np.max(np.abs(audio_16k)):.6f}")


if __name__ == "__main__":
    main()
