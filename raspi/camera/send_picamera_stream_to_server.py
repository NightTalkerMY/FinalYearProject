# Import required modules for WebRTC, Pi Camera video capture, audio capture, and HTTP
import os
import asyncio
import aiohttp
import av
import numpy as np
import sounddevice as sd
from picamera2 import Picamera2
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
    VideoStreamTrack,
    AudioStreamTrack
)

# === Static Configuration ===
FRAME_WIDTH  = 960
FRAME_HEIGHT = 540
FRAME_RATE   = 30

AUDIO_SAMPLE_RATE = 48000  # WebRTC expects 48kHz
AUDIO_CHANNELS = 1
AUDIO_FRAME_DURATION = 0.020  # 20ms per audio frame (WebRTC standard)
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION)  # 960

SERVER_IP = "100.100.155.39"  # Server IP address
SERVER_PORT = "8889"
MediaMTX_ENDPOINT = "cam1"


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
            # indata shape: (frames, channels) - float32
            audio_data = indata[:, 0].copy()
            try:
                self._loop.call_soon_threadsafe(self.queue.put_nowait, audio_data)
            except Exception:
                pass

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            blocksize=AUDIO_SAMPLES_PER_FRAME,
            callback=callback,
        )
        self.stream.start()
        print(f"[INFO] USB Mic capture started at {AUDIO_SAMPLE_RATE}Hz")

    async def recv(self):
        pts, time_base = await self.next_timestamp()

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
        audio_frame.pts = pts
        audio_frame.time_base = time_base

        return audio_frame

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[INFO] USB Mic capture stopped.")


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

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)

    # Attach video track
    video_track = PiCameraVideoStreamTrack()
    pc.addTrack(video_track)

    # Attach audio track (if mic available)
    if audio_track:
        pc.addTrack(audio_track)
        audio_track.start_capture(asyncio.get_event_loop())

    # Event to know when connection is dead
    disconnect_event = asyncio.Event()

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[STATE] pc.connectionState = {pc.connectionState}")
        if pc.connectionState in ("failed", "disconnected", "closed"):
            disconnect_event.set()

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    print("[INFO] SDP offer created successfully (video + audio)")

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
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer")
            )
            print("[SUCCESS] WebRTC connection established with MediaMTX! (video + audio)")

    # Wait until the connection dies
    try:
        await disconnect_event.wait()
        print("[WARN] Connection ended or failed, cleaning up this attempt...")
    finally:
        await pc.close()
        video_track.picam2.stop()
        if audio_track:
            audio_track.stop()
        print("[INFO] Stream closed. Camera and mic released.")

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Stopped by user")
            
