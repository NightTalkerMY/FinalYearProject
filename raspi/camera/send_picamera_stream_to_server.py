# Import required modules for WebRTC, Pi Camera video capture, audio capture, and HTTP
import os
import sys
import time
import asyncio
import fractions
import threading
import aiohttp
import av
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from picamera2 import Picamera2
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
    VideoStreamTrack,
    AudioStreamTrack
)
from openwakeword.model import Model as OWWModel

# === Static Configuration ===
FRAME_WIDTH  = 960
FRAME_HEIGHT = 540
FRAME_RATE   = 30

AUDIO_SAMPLE_RATE = 48000  # WebRTC expects 48kHz
AUDIO_CHANNELS = 1
AUDIO_FRAME_DURATION = 0.020  # 20ms per audio frame (WebRTC standard)
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION)  # 960

SERVER_IP = os.getenv("HOLOPI_BACKEND_HOST", "100.100.155.39")
SERVER_PORT = os.getenv("HOLOPI_MEDIAMTX_PORT", "8889")
MediaMTX_ENDPOINT = os.getenv("HOLOPI_MEDIAMTX_INPUT_STREAM", "cam1")

# Wake word config
OWW_CHUNK_SIZE = 1280       # 80ms at 16kHz
OWW_TARGET_RATE = 16000
OWW_THRESHOLD = 0.5
OWW_COOLDOWN_SEC = 2.0
MAX_LISTEN_SEC = 15.0       # Hard safety timeout (desktop should send stop before this)
PI_LISTEN_PORT = int(os.getenv("HOLOPI_PI_LISTEN_PORT", "5051"))
DESKTOP_WAKEWORD_URL = os.getenv(
    "HOLOPI_DESKTOP_WAKEWORD_URL",
    f"http://{SERVER_IP}:5050/wakeword",
)
DESKTOP_UTTERANCE_URL = os.getenv(
    "HOLOPI_DESKTOP_UTTERANCE_URL",
    f"http://{SERVER_IP}:5050/utterance",
)
# Set to onnx model filename (looked up in same dir as this script), or None for built-in
OWW_MODEL_PATH = os.getenv("HOLOPI_WAKEWORD_MODEL", "hey_holo.onnx") or None


# === USB Mic Detection (from pi_ear.py) ===
def get_usb_mic_index():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
            return i
    return None


# === Custom Video Track for Pi Camera ===
class PiCameraVideoStreamTrack(VideoStreamTrack):
    """
    Custom video track to capture frames from Raspberry Pi Camera using picamera2.
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        print("[INFO] Initializing Pi Camera...")

        self.picam2 = Picamera2()

        config = self.picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT)},
            controls={
                "FrameRate": FRAME_RATE,
                "NoiseReductionMode": 1,
                "Sharpness": 1.2
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        print(f"[INFO] Pi Camera started at {FRAME_WIDTH}x{FRAME_HEIGHT}@{FRAME_RATE}fps")

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self.picam2.capture_array()

        frame_rgb = np.ascontiguousarray(frame[..., :3])
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame


# === Custom Audio Track for USB Microphone ===
class UsbMicAudioStreamTrack(AudioStreamTrack):
    """
    Custom audio track that captures from a USB microphone using sounddevice
    and delivers 20ms frames at 48kHz mono for WebRTC.
    """
    kind = "audio"

    def __init__(self, device_index):
        super().__init__()
        self.device_index = device_index
        self.queue = asyncio.Queue()
        self.stream = None
        self._loop = None
        print(f"[INFO] USB Mic audio track created (device {device_index})")

    def start_capture(self, loop):
        """Start the sounddevice InputStream. Must be called from the event loop thread."""
        self._loop = loop

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[AUDIO] sounddevice status: {status}")
            # indata shape: (frames, channels) - float32
            audio_data = indata[:, 0].copy()
            try:
                self._loop.call_soon_threadsafe(self.queue.put_nowait, audio_data)
            except Exception:
                pass
            # Feed wake word detector with clean 48kHz audio
            if hasattr(self, 'ww_detector') and self.ww_detector:
                self.ww_detector.feed_audio(audio_data)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                blocksize=AUDIO_SAMPLES_PER_FRAME,
                callback=callback,
            )
            self.stream.start()
            print(f"[INFO] USB Mic capture started at {AUDIO_SAMPLE_RATE}Hz")
        except Exception as e:
            print(f"[ERROR] Failed to start USB Mic capture: {e}")
            self.stream = None

    async def recv(self):
        # AudioStreamTrack does NOT have next_timestamp() — only VideoStreamTrack does.
        # We manage our own timing, matching the base AudioStreamTrack pattern.
        from aiortc.mediastreams import MediaStreamError

        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += AUDIO_SAMPLES_PER_FRAME
            wait = self._start + (self._timestamp / AUDIO_SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        # If stream failed to start, return silence
        if self.stream is None:
            samples = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.float32)
        else:
            # Get audio samples from the sounddevice callback queue
            try:
                samples = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Silence if mic stalls
                samples = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.float32)

        # Ensure correct length
        if len(samples) < AUDIO_SAMPLES_PER_FRAME:
            samples = np.pad(samples, (0, AUDIO_SAMPLES_PER_FRAME - len(samples)))
        elif len(samples) > AUDIO_SAMPLES_PER_FRAME:
            samples = samples[:AUDIO_SAMPLES_PER_FRAME]

        # Convert float32 [-1,1] to int16 for av.AudioFrame
        samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        # Build av.AudioFrame (mono, s16 format, 48kHz)
        audio_frame = av.AudioFrame.from_ndarray(
            samples_int16.reshape(1, -1), format="s16", layout="mono"
        )
        audio_frame.sample_rate = AUDIO_SAMPLE_RATE
        audio_frame.pts = self._timestamp
        audio_frame.time_base = fractions.Fraction(1, AUDIO_SAMPLE_RATE)

        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count <= 3 or self._frame_count % 500 == 0:
            rms = np.sqrt(np.mean(samples ** 2) + 1e-12)
            print(f"[AUDIO] frame={self._frame_count} rms={rms:.6f} pts={self._timestamp}")
            sys.stdout.flush()

        return audio_frame

    def stop(self):
        import traceback
        print("[INFO] USB Mic capture stop() called. Stack trace:")
        traceback.print_stack()
        sys.stdout.flush()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[INFO] USB Mic capture stopped.")


# === Wake Word Detector (runs on Pi with clean mic audio) ===
class WakeWordDetector:
    """
    OWW detection on clean mic audio.
    On wake word: notify desktop, record until desktop sends /stop (or hard timeout).
    Then send clean recorded audio to desktop for STT.
    """

    def __init__(self):
        if OWW_MODEL_PATH:
            model_path = OWW_MODEL_PATH
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.path.dirname(__file__), model_path)
            print(f"[OWW] Loading model: {model_path}")
            try:
                self.model = OWWModel(wakeword_models=[model_path])
            except TypeError:
                self.model = OWWModel([model_path])
        else:
            self.model = OWWModel()
        print(f"[OWW] Wake words: {list(self.model.models.keys())}")

        self.accum = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        self.count = 0

        # State: IDLE / LISTENING
        self.state = "IDLE"
        self.cooldown_until = 0
        self.listen_start_time = 0
        self.utter_frames = []

        # Stop signal from desktop (set via HTTP /stop)
        self.stop_event = threading.Event()

        # Start HTTP server to receive /stop from desktop
        self._start_stop_server()

    def _start_stop_server(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        detector = self

        class StopHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/stop":
                    detector.stop_event.set()
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        self._server = HTTPServer(("0.0.0.0", PI_LISTEN_PORT), StopHandler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[OWW] Listening for /stop on port {PI_LISTEN_PORT}")

    def feed_audio(self, audio_f32_48k):
        """Called from the mic callback with 48kHz float32 audio."""
        audio_16k = resample_poly(audio_f32_48k, 1, 3)

        with self.lock:
            self.accum = np.concatenate((self.accum, audio_16k))

        while True:
            with self.lock:
                if len(self.accum) < OWW_CHUNK_SIZE:
                    break
                chunk = self.accum[:OWW_CHUNK_SIZE]
                self.accum = self.accum[OWW_CHUNK_SIZE:]

            chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            now = time.time()

            if self.state == "IDLE":
                prediction = self.model.predict(chunk_int16)
                self.count += 1

                if self.count % 25 == 0:
                    top = max(prediction.items(), key=lambda x: x[1])
                    print(f"  [OWW #{self.count:4d}] {top[0]}={top[1]:.4f}")
                    sys.stdout.flush()

                if now < self.cooldown_until:
                    continue

                for name, score in prediction.items():
                    if score >= OWW_THRESHOLD:
                        print(f"[OWW] WAKE '{name}' score={score:.2f} -> RECORDING")
                        sys.stdout.flush()
                        self.state = "LISTENING"
                        self.utter_frames = []
                        self.listen_start_time = now
                        self.stop_event.clear()
                        if hasattr(self.model, 'reset'):
                            self.model.reset()
                        # Notify desktop: wake word fired
                        self._post_async(DESKTOP_WAKEWORD_URL,
                                         json={"word": str(name), "score": float(score)})
                        break

            elif self.state == "LISTENING":
                self.utter_frames.append(chunk)

                is_stop = self.stop_event.is_set()
                is_timeout = (now - self.listen_start_time) >= MAX_LISTEN_SEC

                if is_stop or is_timeout:
                    reason = "ASD stop" if is_stop else "timeout"
                    duration = now - self.listen_start_time
                    print(f"[OWW] END ({reason}, {duration:.1f}s) -> sending audio")
                    sys.stdout.flush()

                    # Send clean audio to desktop
                    if self.utter_frames:
                        audio_f32 = np.concatenate(self.utter_frames).astype(np.float32)
                        audio_int16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
                        self._post_async(DESKTOP_UTTERANCE_URL, data=audio_int16.tobytes())

                    self.utter_frames = []
                    self.stop_event.clear()
                    self.cooldown_until = now + OWW_COOLDOWN_SEC
                    if hasattr(self.model, 'reset'):
                        self.model.reset()
                    self.state = "IDLE"

    def _post_async(self, url, **kwargs):
        def _do():
            import requests
            try:
                requests.post(url, timeout=20, **kwargs)
                print(f"[OWW] Sent -> {url.split('/')[-1]}")
            except Exception as e:
                print(f"[OWW] Failed -> {url.split('/')[-1]}: {e}")
            sys.stdout.flush()
        threading.Thread(target=_do, daemon=True).start()


# === WebRTC Streaming Function ===
async def publish_stream():
    print("[INFO] Preparing WebRTC connection to MediaMTX...")

    # Detect USB mic
    mic_index = get_usb_mic_index()
    if mic_index is None:
        print("[WARN] USB Mic not found! Streaming video only.")
        audio_track = None
    else:
        print(f"[INFO] Found USB Mic at device index {mic_index}")
        audio_track = UsbMicAudioStreamTrack(mic_index)

    # Start wake word detector (taps into same mic audio)
    ww_detector = WakeWordDetector()
    if audio_track:
        audio_track.ww_detector = ww_detector

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)

    # Attach video track
    video_track = PiCameraVideoStreamTrack()
    pc.addTrack(video_track)

    # Attach audio track (if mic available)
    if audio_track:
        pc.addTrack(audio_track)
        audio_track.start_capture(asyncio.get_running_loop())

    # Event to know when connection is dead
    disconnect_event = asyncio.Event()

    @pc.on("connectionstatechange")
    async def on_state_change():
        state = pc.connectionState
        print(f"[STATE] pc.connectionState = {state}")
        # Only treat "failed" and "closed" as terminal.
        # "disconnected" is transient — WebRTC may recover from it.
        if state in ("failed", "closed"):
            disconnect_event.set()

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Log SDP to verify audio is included
    sdp_lines = pc.localDescription.sdp.split('\n')
    audio_lines = [l for l in sdp_lines if 'm=audio' in l]
    video_lines = [l for l in sdp_lines if 'm=video' in l]
    print(f"[INFO] SDP offer: {len(video_lines)} video, {len(audio_lines)} audio m-lines")
    for l in audio_lines:
        print(f"  [SDP] {l.strip()}")
    sys.stdout.flush()

    whip_url = f"http://{SERVER_IP}:{SERVER_PORT}/{MediaMTX_ENDPOINT}/whip"
    print(f"[INFO] Sending offer to WHIP endpoint: {whip_url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            whip_url,
            data=pc.localDescription.sdp,
            headers={"Content-Type": "application/sdp"},
        ) as resp:
            if resp.status != 201:
                print(f"[ERROR] WHIP connection failed: HTTP {resp.status}")
                print(await resp.text())
                await pc.close()
                video_track.picam2.stop()
                if audio_track:
                    audio_track.stop()
                return

            answer_sdp = await resp.text()

            # Log answer SDP to check if audio was accepted
            ans_audio = [l for l in answer_sdp.split('\n') if 'm=audio' in l]
            print(f"[INFO] SDP answer: {len(ans_audio)} audio m-lines")
            for l in ans_audio:
                print(f"  [SDP-ANS] {l.strip()}")
            sys.stdout.flush()

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer")
            )
            print("[SUCCESS] WebRTC connection established with MediaMTX!")

    # Wait until the connection dies
    print("[INFO] Waiting on disconnect_event...")
    sys.stdout.flush()
    try:
        await disconnect_event.wait()
        print(f"[WARN] disconnect_event fired (state={pc.connectionState}), cleaning up...")
        sys.stdout.flush()
    except asyncio.CancelledError:
        print("[WARN] publish_stream was cancelled (likely process terminating)")
        sys.stdout.flush()
    finally:
        await pc.close()
        video_track.picam2.stop()
        if audio_track:
            audio_track.stop()
        print("[INFO] Stream closed. Camera and mic released.")
        sys.stdout.flush()

async def main():
    while True:
        try:
            await publish_stream()
        except Exception as e:
            print(f"[FATAL] Unhandled exception: {e}")

        print("[INFO] Waiting 3 seconds, then trying again..")
        await asyncio.sleep(3)

# === Entry Point ===
if __name__ == "__main__":
    # Check Opus codec availability (required for WebRTC audio)
    try:
        _codec = av.Codec('libopus', 'w')
        print(f"[INFO] Opus encoder available: {_codec.name}")
    except Exception:
        try:
            _codec = av.Codec('opus', 'w')
            print(f"[INFO] Opus encoder available: {_codec.name}")
        except Exception:
            print("[WARN] Opus encoder NOT found in PyAV! Audio may not work.")
            print("[WARN] Try: pip install av --force-reinstall")
    sys.stdout.flush()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Stopped by user")
            
