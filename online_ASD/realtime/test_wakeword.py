"""
Test script: Run openwakeword on a WAV file or live from WHEP, printing scores.
Usage:
  python test_wakeword.py                          # live from WHEP
  python test_wakeword.py test_pi_audio.wav        # from WAV file
  python test_wakeword.py --model hey_holo.onnx    # use custom model
  python test_wakeword.py test_pi_audio.wav --model hey_holo.onnx
"""
import sys
import os
import wave
import numpy as np
from openwakeword.model import Model as OWWModel

# Parse args
wav_file = None
custom_model = None
args = sys.argv[1:]
while args:
    if args[0] == "--model":
        custom_model = args[1]
        args = args[2:]
    else:
        wav_file = args[0]
        args = args[1:]

CHUNK_SIZE = 1280  # 80ms at 16kHz (what openwakeword expects)
SAMPLE_RATE = 16000


def load_model():
    if custom_model:
        model_path = custom_model
        if not os.path.isabs(model_path):
            # Check in wakeword_models/ directory
            alt_path = os.path.join(os.path.dirname(__file__), "wakeword_models", model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
        print(f"Loading custom model: {model_path}")
        model = OWWModel(wakeword_models=[model_path])
    else:
        print("Loading built-in models (hey_jarvis, etc.)...")
        model = OWWModel()

    print(f"Available wake words: {list(model.models.keys())}\n")
    return model


def test_from_wav(model, filepath):
    print(f"Reading: {filepath}")
    with wave.open(filepath, 'r') as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
        assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()*8}-bit"

        n_frames = wf.getnframes()
        duration = n_frames / SAMPLE_RATE
        print(f"Duration: {duration:.1f}s, {n_frames} samples\n")

        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16)

    # Process in chunks
    chunk_idx = 0
    max_scores = {}

    for i in range(0, len(audio) - CHUNK_SIZE, CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        prediction = model.predict(chunk)

        chunk_idx += 1
        time_sec = i / SAMPLE_RATE

        # Track max scores
        for name, score in prediction.items():
            if name not in max_scores or score > max_scores[name]:
                max_scores[name] = score

        # Print every 10th prediction or if any score > 0.1
        any_high = any(s > 0.1 for s in prediction.values())
        if chunk_idx % 10 == 0 or any_high:
            scores = "  ".join(f"{k}={v:.4f}" for k, v in prediction.items())
            marker = " <<<" if any_high else ""
            print(f"  t={time_sec:5.1f}s  {scores}{marker}")

    print(f"\n--- Max scores ---")
    for name, score in max_scores.items():
        status = "WOULD TRIGGER" if score >= 0.5 else "no trigger"
        print(f"  {name}: {score:.4f} ({status})")


def test_from_whep(model):
    """Live test: connect to WHEP and print scores in real-time."""
    import asyncio
    import aiohttp
    from scipy.signal import resample_poly
    from aiortc import RTCPeerConnection, RTCSessionDescription

    WHEP_URL = "http://127.0.0.1:8889/cam1/whep"
    accum = np.array([], dtype=np.float32)
    chunk_count = [0]
    max_scores = {}

    async def run():
        nonlocal accum
        pc = RTCPeerConnection()
        pc.addTransceiver("audio", direction="recvonly")

        audio_ready = asyncio.Event()
        track_ref = [None]

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                track_ref[0] = track
                audio_ready.set()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                async with session.post(
                    WHEP_URL,
                    data=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"}
                ) as resp:
                    if resp.status == 201:
                        answer = await resp.text()
                        break
                    await asyncio.sleep(2)
            else:
                print("[FAIL] Could not connect")
                return

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        await audio_ready.wait()
        print(f"[OK] Connected! Say your wake word... (Ctrl+C to stop)\n")

        track = track_ref[0]
        try:
            while True:
                frame = await asyncio.wait_for(track.recv(), timeout=3.0)
                samples = frame.to_ndarray().flatten().astype(np.float32)

                if np.max(np.abs(samples)) > 2.0:
                    samples = samples / 32768.0

                if frame.sample_rate != SAMPLE_RATE:
                    ratio = frame.sample_rate // SAMPLE_RATE
                    if ratio > 1:
                        samples = resample_poly(samples, 1, ratio)

                accum = np.concatenate((accum, samples))

                while len(accum) >= CHUNK_SIZE:
                    chunk_f32 = accum[:CHUNK_SIZE]
                    accum = accum[CHUNK_SIZE:]

                    chunk_int16 = (np.clip(chunk_f32, -1.0, 1.0) * 32767).astype(np.int16)
                    prediction = model.predict(chunk_int16)
                    chunk_count[0] += 1

                    for name, score in prediction.items():
                        if name not in max_scores or score > max_scores[name]:
                            max_scores[name] = score

                    any_high = any(s > 0.1 for s in prediction.values())
                    if chunk_count[0] % 12 == 0 or any_high:
                        rms = np.sqrt(np.mean(chunk_f32 ** 2) + 1e-12)
                        scores = "  ".join(f"{k}={v:.4f}" for k, v in prediction.items())
                        marker = " <<<" if any_high else ""
                        print(f"  #{chunk_count[0]:4d} rms={rms:.4f}  {scores}{marker}")

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await pc.close()

        print(f"\n--- Max scores ({chunk_count[0]} predictions) ---")
        for name, score in max_scores.items():
            status = "WOULD TRIGGER" if score >= 0.5 else "no trigger"
            print(f"  {name}: {score:.4f} ({status})")

    asyncio.run(run())


if __name__ == "__main__":
    model = load_model()

    if wav_file:
        test_from_wav(model, wav_file)
    else:
        print("No WAV file specified — running live from WHEP\n")
        test_from_whep(model)
