"""
Integrated ASD + Wake Word + STT Pipeline
==========================================
Consumes video+audio from MediaMTX via WHEP and runs Active Speaker Detection.
Wake word detection runs on the Pi (clean mic audio via openwakeword).
ASD gates and controls the recording session: confirms a visible speaker (start),
detects when they stop (stop signal to Pi), then receives clean audio for STT.

Flow: IDLE -> /wakeword -> CONFIRMING -> (ASD green) -> LISTENING -> (ASD red ~1s) -> /stop -> PROCESSING -> STT -> IDLE
"""

import os
import sys
import cv2
import enum
import time
import queue
import asyncio
import threading
import requests

import torch
import torch.nn as nn
import numpy as np
import python_speech_features
import aiohttp
from collections import deque
from scipy.signal import resample_poly
from aiortc import RTCPeerConnection, RTCSessionDescription
# OWW now runs on Pi — desktop receives wake word signal via HTTP

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Path setup (same pattern as realtime_main.py) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.online_tracker import OnlineTracker
from model.faceDetector.s3fd import S3FD
from utils.student_model import CausalStudentASD
from ASD import ASD

# ==========================================
# CONFIGURATION
# ==========================================
WHEP_URL = os.getenv("HOLOPI_CAM1_WHEP_URL", "http://127.0.0.1:8889/cam1/whep")
STT_URL = os.getenv("HOLOPI_STT_URL", "http://127.0.0.1:8000/transcribe")
ORCHESTRATOR_URL = os.getenv(
    "HOLOPI_ASD_ORCHESTRATOR_URL",
    "http://127.0.0.1:5000/process_text",
)

WAKEWORD_LISTEN_PORT = int(os.getenv("HOLOPI_WAKEWORD_LISTEN_PORT", "5050"))
PI_HOST = os.getenv("HOLOPI_PI_HOST", "100.80.70.120")
PI_STOP_URL = os.getenv("HOLOPI_PI_STOP_URL", f"http://{PI_HOST}:5051/stop")

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PRETRAIN_DIR = os.path.join(REPOSITORY_ROOT, 'online_ASD', 'pretrain_model')
configured_student_weights = os.getenv(
    "HOLOPI_CATT_STUDENT_WEIGHTS",
    os.path.join(PRETRAIN_DIR, 'SOTA_studen_model', 'holopi_student_best.pt'),
)
STUDENT_WEIGHTS = (
    configured_student_weights
    if os.path.isabs(configured_student_weights)
    else os.path.join(REPOSITORY_ROOT, configured_student_weights)
)

ACTUAL_FPS = 25.0
SENSITIVITY = 0.4
DETECT_INTERVAL = 5
INFERENCE_INTERVAL = 6
WINDOW_SIZE = 50
MAX_SPEAKERS = 5
AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE / ACTUAL_FPS)  # 640





# ==========================================
# WHEP VIDEO + AUDIO CAPTURE
# ==========================================
class WhepVideoAudioCapture:
    """
    Connects to a MediaMTX WHEP endpoint and consumes both video and audio tracks.
    Video frames go to a queue; audio samples go to a thread-safe ring buffer.
    """

    def __init__(self, url):
        self.url = url
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.pc = None

        # Audio ring buffer (thread-safe)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.total_samples_received = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print(f"[WHEP] Connecting to {self.url}...")
        time.sleep(5)  # Give WHEP time to connect (with retry)

    def get_frame(self):
        """Get the next video frame (non-blocking, returns None if empty)."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def get_audio_chunk(self, num_samples):
        """Get the last N audio samples (for per-frame ASD feeding)."""
        with self.audio_lock:
            if len(self.audio_buffer) < num_samples:
                return np.zeros(num_samples, dtype=np.float32)
            return self.audio_buffer[-num_samples:].copy()

    def get_audio_since(self, start_position):
        """Extract all audio from a given sample position to now (for utterance extraction)."""
        with self.audio_lock:
            samples_available = self.total_samples_received - start_position
            if samples_available <= 0:
                return np.array([], dtype=np.float32)
            samples_available = min(samples_available, len(self.audio_buffer))
            return self.audio_buffer[-samples_available:].copy()

    def get_current_position(self):
        """Get current audio buffer write position (monotonic sample counter)."""
        with self.audio_lock:
            return self.total_samples_received

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def _run_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consume_stream())

    async def _consume_stream(self):
        max_retries = 10
        for attempt in range(max_retries):
            self.pc = RTCPeerConnection()
            self.pc.addTransceiver("video", direction="recvonly")
            self.pc.addTransceiver("audio", direction="recvonly")

            @self.pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    print("[WHEP] Video track received!")
                    asyncio.ensure_future(self._read_video_track(track))
                elif track.kind == "audio":
                    print("[WHEP] Audio track received!")
                    asyncio.ensure_future(self._read_audio_track(track))

            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        data=self.pc.localDescription.sdp,
                        headers={"Content-Type": "application/sdp"}
                    ) as resp:
                        if resp.status != 201:
                            error_text = await resp.text()
                            print(f"[WHEP] Error: {resp.status} - {error_text} (attempt {attempt+1}/{max_retries})")
                            await self.pc.close()
                            await asyncio.sleep(2)
                            continue
                        answer = await resp.text()
            except Exception as e:
                print(f"[WHEP] Connection failed: {e} (attempt {attempt+1}/{max_retries})")
                await self.pc.close()
                await asyncio.sleep(2)
                continue

            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer, type="answer")
            )
            print("[WHEP] Connected successfully!")
            break
        else:
            print("[WHEP] Failed to connect after all retries!")
            return

        while self.running:
            await asyncio.sleep(1)

        await self.pc.close()

    async def _read_video_track(self, track):
        while self.running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(img)
            except Exception:
                break

    async def _read_audio_track(self, track):
        frame_count = 0
        while self.running:
            try:
                frame = await track.recv()
                # av.AudioFrame -> numpy. Shape is (channels, samples) for planar formats
                samples = frame.to_ndarray().flatten().astype(np.float32)

                # Debug: log first few audio frames
                if frame_count < 3:
                    print(f"[WHEP] Audio frame {frame_count}: {samples.shape}, "
                          f"rate={frame.sample_rate}, max={np.max(np.abs(samples)):.4f}")
                frame_count += 1

                # Normalize int16 range to float32 [-1, 1] if needed
                if np.max(np.abs(samples)) > 2.0:
                    samples = samples / 32768.0

                # Resample to 16kHz if source is different (typically 48kHz from WebRTC)
                if frame.sample_rate != AUDIO_SAMPLE_RATE:
                    ratio = frame.sample_rate // AUDIO_SAMPLE_RATE
                    if ratio > 1:
                        samples = resample_poly(samples, 1, ratio)

                with self.audio_lock:
                    self.audio_buffer = np.concatenate((self.audio_buffer, samples))
                    self.total_samples_received += len(samples)
                    # Keep last ~10 seconds to prevent RAM overflow
                    max_buf = AUDIO_SAMPLE_RATE * 10
                    if len(self.audio_buffer) > max_buf:
                        self.audio_buffer = self.audio_buffer[-max_buf:]
            except Exception as e:
                print(f"[WHEP] Audio track ended: {e}")
                break
        print(f"[WHEP] Audio reader exited after {frame_count} frames")


# ==========================================
# MASTER TRANSLATOR (copied from realtime_main.py to avoid side-effect imports)
# ==========================================
class MasterTranslator(nn.Module):
    def __init__(self, pretrain_dir):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = (self.device.type == 'cuda')

        self.cue_configs = [
            {'name': 'face',            'path': 'face_model',             'size': 112},
            {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
            {'name': 'face_body',       'path': 'body_model',             'size': 224},
            {'name': 'face_large',      'path': 'face_large_model',       'size': 112},
            {'name': 'background',      'path': 'background_model.model', 'size': 224},
            {'name': 'face_small',      'path': 'face_small_model',       'size': 112},
            {'name': 'face_down',       'path': 'face_down_model',        'size': 112}
        ]

        self.visual_encoders = nn.ModuleDict()
        self.audio_encoder = None

        print("--- Booting Master Translator (Loading 7 CNNs, fp16=%s) ---" % self.use_fp16)
        for config in self.cue_configs:
            name = config['name']
            full_path = os.path.join(pretrain_dir, config['path'])
            s = ASD(encoder_struct=[32, 64, 128])
            if os.path.isdir(full_path):
                s.loadParameters_multi(full_path)
            else:
                s.loadParameters(full_path)

            s.eval()
            encoder = s.model.visualEncoder.to(self.device)
            if self.use_fp16:
                encoder = encoder.half()
            self.visual_encoders[name] = encoder
            if name == 'face':
                audio_enc = s.model.audioEncoder.to(self.device)
                if self.use_fp16:
                    audio_enc = audio_enc.half()
                self.audio_encoder = audio_enc

    def forward_visual(self, crops_dict):
        visual_vectors = []
        with torch.inference_mode():
            for config in self.cue_configs:
                name = config['name']
                crop_seq = crops_dict[name]

                crop_tensor = torch.FloatTensor(crop_seq).unsqueeze(0).unsqueeze(0).to(self.device)
                crop_tensor = (crop_tensor / 255.0 - 0.4161) / 0.1688
                if self.use_fp16:
                    crop_tensor = crop_tensor.half()

                v_feat = self.visual_encoders[name](crop_tensor)
                visual_vectors.append(v_feat.squeeze(0).float().cpu().numpy())

        return np.concatenate(visual_vectors, axis=-1)  # [50, 896]

    def forward_audio(self, continuous_audio):
        with torch.inference_mode():
            a_tensor = torch.FloatTensor(continuous_audio).unsqueeze(0).to(self.device)
            a_tensor = a_tensor.unsqueeze(1).transpose(2, 3)
            if self.use_fp16:
                a_tensor = a_tensor.half()
            a_feat = self.audio_encoder(a_tensor)
        return a_feat.squeeze(0).float().cpu().numpy()  # [50, 128]


def get_crop(img_gray, x1, y1, x2, y2, size):
    h, w = img_gray.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = img_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(crop, (size, size))


# ==========================================
# PIPELINE STATE MACHINE
# ==========================================
class PipelineState(enum.Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"   # Wake word fired, waiting for ASD green
    LISTENING = "LISTENING"     # ASD confirmed, Pi is recording
    PROCESSING = "PROCESSING"  # Stop sent to Pi, waiting for utterance


SILENCE_FRAMES_THRESHOLD = 25  # ~1s at 25fps of ASD red before sending stop
ASD_CONFIRM_TIMEOUT = 5.0     # Max seconds to wait for ASD green after wake word

class PipelineStateMachine:
    """
    ASD-controlled pipeline.

    IDLE -> /wakeword -> CONFIRMING (wait for ASD green)
    CONFIRMING -> ASD green -> LISTENING (Pi is recording)
    CONFIRMING -> timeout -> /stop to Pi, reject -> IDLE
    LISTENING -> ASD red ~1s -> /stop to Pi -> PROCESSING (wait for utterance)
    PROCESSING -> /utterance -> STT -> IDLE
    """
    def __init__(self):
        self.state = PipelineState.IDLE
        self.ww_time = None
        self.consecutive_silence = 0

    def on_wake_word(self, word, score, any_speaker_active):
        if self.state != PipelineState.IDLE:
            return

        self.ww_time = time.time()
        self.consecutive_silence = 0

        if any_speaker_active:
            self.state = PipelineState.LISTENING
            print(f"[STATE] IDLE -> LISTENING (ASD already green)")
        else:
            self.state = PipelineState.CONFIRMING
            print(f"[STATE] IDLE -> CONFIRMING (waiting for ASD green)")

    def on_asd_update(self, any_speaker_active):
        if self.state == PipelineState.CONFIRMING:
            elapsed = time.time() - self.ww_time
            if any_speaker_active:
                self.state = PipelineState.LISTENING
                self.consecutive_silence = 0
                print(f"[STATE] CONFIRMING -> LISTENING (ASD green after {elapsed:.1f}s)")
            elif elapsed > ASD_CONFIRM_TIMEOUT:
                print(f"[STATE] CONFIRMING -> IDLE (ASD never green, sending /stop)")
                self._send_stop()
                self._reset()

        elif self.state == PipelineState.LISTENING:
            if any_speaker_active:
                self.consecutive_silence = 0
            else:
                self.consecutive_silence += 1

            if self.consecutive_silence >= SILENCE_FRAMES_THRESHOLD:
                print(f"[STATE] LISTENING -> PROCESSING (ASD red {SILENCE_FRAMES_THRESHOLD} frames, sending /stop)")
                self._send_stop()
                self.state = PipelineState.PROCESSING

    def on_utterance_received(self):
        if self.state in (PipelineState.PROCESSING, PipelineState.LISTENING):
            print(f"[STATE] {self.state.value} -> sending to STT")
            return True
        else:
            print(f"[STATE] Utterance arrived in {self.state.value}, ignoring")
            return False

    def on_processing_complete(self):
        self._reset()
        print("[STATE] -> IDLE")

    def _send_stop(self):
        def _do():
            try:
                requests.post(PI_STOP_URL, timeout=3)
                print("[STATE] /stop sent to Pi")
            except Exception as e:
                print(f"[STATE] /stop failed: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def _reset(self):
        self.state = PipelineState.IDLE
        self.ww_time = None
        self.consecutive_silence = 0


# ==========================================
# PI LISTENER (receives wake word + clean utterance from Pi)
# ==========================================
class PiReceiver:
    """
    HTTP server receiving from Pi:
      POST /wakeword  — wake word fired (JSON: word, score)
      POST /utterance — clean recorded audio (raw int16 bytes)
    """

    def __init__(self, port=WAKEWORD_LISTEN_PORT):
        self.wakeword_event = threading.Event()
        self.last_word = None
        self.last_score = 0

        self.utterance_audio = None
        self.utterance_ready = threading.Event()

        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                path = self.path
                length = int(self.headers.get('Content-Length', 0))

                if path == "/wakeword":
                    body = json.loads(self.rfile.read(length)) if length else {}
                    receiver.last_word = body.get("word", "?")
                    receiver.last_score = body.get("score", 0)
                    receiver.wakeword_event.set()
                    print(f"[PI] Wake word: '{receiver.last_word}' score={receiver.last_score:.2f}")

                elif path == "/utterance":
                    raw = self.rfile.read(length)
                    receiver.utterance_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                    receiver.utterance_ready.set()
                    dur = len(receiver.utterance_audio) / AUDIO_SAMPLE_RATE
                    print(f"[PI] Utterance received ({dur:.1f}s)")

                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        self.server = HTTPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[PI] Listening on port {port}")

    def check_wakeword(self):
        """Non-blocking. Returns (word, score) if triggered, else None."""
        if self.wakeword_event.is_set():
            self.wakeword_event.clear()
            return self.last_word, self.last_score
        return None

    def check_utterance(self):
        """Non-blocking. Returns float32 audio if ready, else None."""
        if self.utterance_ready.is_set():
            self.utterance_ready.clear()
            audio = self.utterance_audio
            self.utterance_audio = None
            return audio
        return None


# ==========================================
# STT SUBMISSION (runs in background thread)
# ==========================================
def submit_to_stt(audio_float32, stt_url, orchestrator_url):
    """Sends recorded utterance to STT, then forwards transcription to the orchestrator."""
    audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
    raw_bytes = audio_int16.tobytes()

    try:
        stt_resp = requests.post(stt_url, data=raw_bytes, timeout=20)
        text = stt_resp.json().get("text", "")
    except Exception as e:
        print(f"[PIPELINE] STT request failed: {e}")
        return

    if not text.strip():
        print("[PIPELINE] STT returned empty text, skipping.")
        return

    print(f"[PIPELINE] STT result: {text}")

    try:
        requests.post(orchestrator_url, json={"text": text}, timeout=20)
    except Exception as e:
        print(f"[PIPELINE] Orchestrator request failed: {e}")


# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    # --- 1. Initialize capture ---
    capture = WhepVideoAudioCapture(WHEP_URL)
    capture.start()

    # --- 2. Load ASD models ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face_detector = S3FD(device='cuda')
    tracker = OnlineTracker(iou_threshold=0.4, max_lost=10, alpha=0.8)
    master_translator = MasterTranslator(PRETRAIN_DIR).to(device).eval()

    asd_model = CausalStudentASD().to(device)
    asd_model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    asd_model.eval()

    # --- 3. Pi receiver (wake word + utterance from Pi) ---
    pi_receiver = PiReceiver()

    # --- 4. Initialize state ---
    state_machine = PipelineStateMachine()

    track_history = {}
    bbox_history = {}
    display_buffer = deque(maxlen=WINDOW_SIZE)
    global_audio_buffer = deque(maxlen=WINDOW_SIZE)
    ai_state_cache = {}

    # Background inference thread state
    pending_snapshot = [None]
    snapshot_lock = threading.Lock()
    stop_event = threading.Event()
    last_detections = []

    # --- 5. Background ASD inference worker (same pattern as realtime_main.py) ---
    def inference_worker():
        while not stop_event.is_set():
            with snapshot_lock:
                snap = pending_snapshot[0]
                pending_snapshot[0] = None

            if snap is None:
                time.sleep(0.005)
                continue

            audio_data = snap['audio']
            tracks = snap['tracks']

            continuous_raw_audio = np.concatenate(audio_data, axis=0)
            scaled_audio = continuous_raw_audio * 32767.0 if np.max(np.abs(continuous_raw_audio)) <= 2.0 else continuous_raw_audio
            audio_mfcc = python_speech_features.mfcc(
                scaled_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010
            )
            if audio_mfcc.shape[0] > 200:
                audio_mfcc = audio_mfcc[:200, :]
            elif audio_mfcc.shape[0] < 200:
                audio_mfcc = np.pad(audio_mfcc, ((0, 200 - audio_mfcc.shape[0]), (0, 0)), mode='edge')

            seq_a_128 = master_translator.forward_audio(audio_mfcc)

            v_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 896), dtype=np.float32)
            a_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 128), dtype=np.float32)
            s_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 4), dtype=np.float32)
            m_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE), dtype=np.float32)

            speaker_mapping = {}
            slot_idx = 0

            for track_id, track_data in tracks.items():
                if slot_idx >= MAX_SPEAKERS:
                    break
                stacked_crops = {k: np.stack(track_data['raw_v'][k]) for k in track_data['raw_v'].keys()}
                seq_v_896 = master_translator.forward_visual(stacked_crops)

                v_tensor[0, slot_idx] = seq_v_896
                a_tensor[0, slot_idx] = seq_a_128
                s_tensor[0, slot_idx] = np.array(track_data['s'])
                m_tensor[0, slot_idx] = 1.0

                speaker_mapping[slot_idx] = track_id
                slot_idx += 1

            if slot_idx > 0:
                with torch.no_grad():
                    v_t = torch.FloatTensor(v_tensor).to(device)
                    a_t = torch.FloatTensor(a_tensor).to(device)
                    s_t = torch.FloatTensor(s_tensor).to(device)
                    m_t = torch.FloatTensor(m_tensor).to(device)

                    logits, _ = asd_model(v_t, a_t, s_t, m_t)
                    frame_probs = torch.nn.functional.softmax(logits[0, :, -1, :], dim=-1)

                for slot, tid in speaker_mapping.items():
                    new_prob = frame_probs[slot, 1].item()
                    old_prob = ai_state_cache.get(tid, new_prob)
                    if new_prob > old_prob:  # onset: react fast
                        ai_state_cache[tid] = 0.9 * new_prob + 0.1 * old_prob
                    else:                    # offset: decay slowly
                        ai_state_cache[tid] = 0.4 * new_prob + 0.6 * old_prob

    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()

    # --- 6. Main loop ---
    frame_idx = 0
    print("\n[PIPELINE] Starting integrated pipeline... Press 'q' to quit.")

    with torch.no_grad():
        while True:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            display_buffer.append(frame.copy())

            # ---- FACE DETECTION & TRACKING ----
            if frame_idx % DETECT_INTERVAL == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
                last_detections = [det[:-1].tolist() for det in raw_detections]

            active_tracks = tracker.update(last_detections)
            stale_ids = [tid for tid in track_history if tid not in active_tracks]
            for tid in stale_ids:
                del track_history[tid]
                del bbox_history[tid]
                if tid in ai_state_cache:
                    del ai_state_cache[tid]

            # ---- RAW AUDIO FOR ASD ----
            raw_audio = capture.get_audio_chunk(SAMPLES_PER_FRAME)
            global_audio_buffer.append(raw_audio)

            # Status line every ~2s
            if frame_idx % 50 == 0:
                rms = np.sqrt(np.mean(raw_audio ** 2) + 1e-12)
                asd_str = " ".join(f"T{tid}:{'SPK' if p>SENSITIVITY else 'sil'}({p:.2f})" for tid, p in ai_state_cache.items()) or "no-faces"
                pend = ""
                print(f"[f{frame_idx:5d}] {state_machine.state.value:10s} | rms={rms:.4f} | {asd_str}{pend}")

            # ---- VISUAL CROP BUFFERING ----
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            for track_id, bbox in active_tracks.items():
                x1, y1, x2, y2 = bbox

                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = (x2 - x1) * 1.2, (y2 - y1) * 1.2

                col_x1 = cx - bw / 2.0
                col_y1 = cy - bh / 2.0
                col_x2 = cx + bw / 2.0
                col_y2 = cy + bh / 2.0
                fw, fh = bw, bh

                crops = {
                    'face':            get_crop(frame_gray, col_x1, col_y1, col_x2, col_y2, 112),
                    'face_large':      get_crop(frame_gray, col_x1 - 0.25*fw, col_y1 - 0.25*fh, col_x2 + 0.25*fw, col_y2 + 0.25*fh, 112),
                    'face_body':       get_crop(frame_gray, col_x1 - 0.75*fw, col_y1, col_x2 + 0.75*fw, col_y2 + 3.0*fh, 224),
                    'face_body_large': get_crop(frame_gray, col_x1 - 1.1*fw, col_y1 - 0.5*fw, col_x2 + 1.1*fw, col_y2 + 3.0*fh, 224),
                    'face_small':      get_crop(frame_gray, col_x1 + 0.25*fw, col_y1 + 0.4*fh, col_x2 - 0.25*fw, col_y2 - 0.1*fh, 112),
                    'face_down':       get_crop(frame_gray, col_x1, col_y1 + fh/2, col_x2, col_y2, 112),
                }

                bg_frame = frame_gray.copy()
                cv2.rectangle(bg_frame, (int(col_x1), int(col_y1)), (int(col_x2), int(col_y2)), (0, 0, 0), thickness=cv2.FILLED)
                crops['background'] = get_crop(bg_frame, 0, 0, w, h, 224)

                norm_cx = (col_x1 + col_x2) / 2.0 / w
                norm_cy = (col_y1 + col_y2) / 2.0 / h
                norm_bw, norm_bh = fw / w, fh / h
                spatial_4d = np.array([norm_cx, norm_cy, norm_bw, norm_bh], dtype=np.float32)

                if track_id not in track_history:
                    track_history[track_id] = {
                        'raw_v': {k: deque(maxlen=WINDOW_SIZE) for k in crops.keys()},
                        's': deque(maxlen=WINDOW_SIZE)
                    }
                    bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)

                for k in crops.keys():
                    track_history[track_id]['raw_v'][k].append(crops[k])

                track_history[track_id]['s'].append(spatial_4d)
                bbox_history[track_id].append([x1, y1, x2, y2])

            # ---- SUBMIT ASD INFERENCE SNAPSHOT ----
            if len(display_buffer) == WINDOW_SIZE and frame_idx % INFERENCE_INTERVAL == 0:
                snap = {'audio': list(global_audio_buffer), 'tracks': {}}
                for track_id, history in track_history.items():
                    if len(history['s']) == WINDOW_SIZE:
                        snap['tracks'][track_id] = {
                            'raw_v': {k: list(history['raw_v'][k]) for k in history['raw_v']},
                            's': list(history['s'])
                        }
                if snap['tracks']:
                    with snapshot_lock:
                        pending_snapshot[0] = snap

            # ---- WAKE WORD CHECK (from Pi via HTTP) ----
            ww_result = pi_receiver.check_wakeword()
            if ww_result:
                ww_name, ww_score = ww_result
                any_active = any(p > SENSITIVITY for p in ai_state_cache.values())
                state_machine.on_wake_word(ww_name, ww_score, any_active)

            # ---- ASD STATE UPDATE ----
            any_speaker_active = any(p > SENSITIVITY for p in ai_state_cache.values())
            state_machine.on_asd_update(any_speaker_active)

            # ---- UTTERANCE FROM PI (clean audio) ----
            utterance = pi_receiver.check_utterance()
            if utterance is not None:
                if state_machine.on_utterance_received():
                    dur = len(utterance) / AUDIO_SAMPLE_RATE
                    print(f"[PIPELINE] Sending {dur:.1f}s clean audio to STT")
                    state_machine.on_processing_complete()
                    threading.Thread(
                        target=submit_to_stt,
                        args=(utterance, STT_URL, ORCHESTRATOR_URL),
                        daemon=True
                    ).start()

            # ---- DISPLAY ----
            if len(display_buffer) == WINDOW_SIZE:
                display_frame = display_buffer[-1].copy()

                for track_id, bbox_list in bbox_history.items():
                    if len(bbox_list) > 0:
                        prob_speaking = ai_state_cache.get(track_id, 0.0)
                        is_speaking = prob_speaking > SENSITIVITY

                        tx1, ty1, tx2, ty2 = map(int, bbox_list[-1])
                        color = (0, 255, 0) if is_speaking else (0, 0, 255)
                        label = f"ID:{track_id} Spk:{prob_speaking:.2f}" if is_speaking else f"ID:{track_id} Sil"

                        cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), color, 3)
                        cv2.putText(display_frame, label, (tx1, max(30, ty1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Pipeline state overlay
                state_text = f"Pipeline: {state_machine.state.value}"
                state_color = {
                    PipelineState.IDLE: (200, 200, 200),
                    PipelineState.CONFIRMING: (0, 165, 255),
                    PipelineState.LISTENING: (0, 255, 255),
                    PipelineState.PROCESSING: (255, 0, 255),
                }[state_machine.state]
                cv2.putText(display_frame, state_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

                cv2.imshow("ASD Pipeline", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    # --- Cleanup ---
    stop_event.set()
    inference_thread.join(timeout=2)
    capture.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
