import cv2
import socket
import struct
import pickle
import numpy as np
import threading
from scipy.io import wavfile

# ==========================================
# VIDEO STREAMERS
# ==========================================

class RemoteVideoClient:
    """Pulls live video frames from the Tailscale remote node."""
    def __init__(self, ip="100.119.215.76", port=9999):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[Video] Connecting to laptop at {ip}:{port} ...")
        self.client.connect((ip, port))
        print("[Video] Connected.")
        self.payload_size = struct.calcsize("Q")

    def _recv_exact(self, size):
        data = b""
        while len(data) < size:
            packet = self.client.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def get_frame(self):
        """Fetches the next frame from the TCP socket."""
        packed_msg_size = self._recv_exact(self.payload_size)
        if packed_msg_size is None:
            return None
            
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        frame_data = self._recv_exact(msg_size)
        if frame_data is None:
            return None
            
        data = pickle.loads(frame_data)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return frame
        
    def close(self):
        self.client.close()


class FileVideoClient:
    """Pulls video frames from a local MP4 file (for offline evaluation)."""
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        print(f"[Video] Loaded local file: {filepath}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def close(self):
        self.cap.release()


# ==========================================
# AUDIO STREAMERS
# ==========================================

class RemoteAudioClient:
    """Pulls live audio from Tailscale and maintains a rolling RAM buffer."""
    def __init__(self, ip="100.119.215.76", port=9998):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[Audio] Connecting to laptop at {ip}:{port} ...")
        self.client.connect((ip, port))
        print("[Audio] Connected.")
        self.payload_size = struct.calcsize("Q")
        
        # Thread-safe rolling buffer for audio
        self.audio_buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        self.running = True
        
        # Start background thread to constantly catch audio packets
        self.thread = threading.Thread(target=self._update_buffer, daemon=True)
        self.thread.start()

    def _recv_exact(self, size):
        data = b""
        while len(data) < size:
            packet = self.client.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _update_buffer(self):
        """Background thread loop that constantly receives audio chunks."""
        try:
            while self.running:
                packed_msg_size = self._recv_exact(self.payload_size)
                if not packed_msg_size:
                    break
                    
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                frame_data = self._recv_exact(msg_size)
                chunk = pickle.loads(frame_data).astype(np.float32)
                
                with self.lock:
                    self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                    
                    # Prevent RAM overflow (keep only the last ~10 seconds at 16000Hz)
                    if len(self.audio_buffer) > 160000:
                        self.audio_buffer = self.audio_buffer[-160000:]
        except Exception as e:
            print(f"[Audio] Connection closed or error: {e}")

    def get_latest_audio(self, num_samples):
        """Extracts the exact number of samples needed for the sliding window."""
        with self.lock:
            if len(self.audio_buffer) < num_samples:
                # Not enough audio has arrived over TCP yet
                return None 
            return self.audio_buffer[-num_samples:]
            
    def close(self):
        self.running = False
        self.client.close()


class FileAudioClient:
    """Pulls audio from a local WAV file and simulates a live stream."""
    def __init__(self, filepath, sample_rate=16000):
        sr, audio = wavfile.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0] # Convert stereo to mono
        self.audio_data = audio
        self.sample_rate = sample_rate
        print(f"[Audio] Loaded local file: {filepath} ({sr}Hz)")

    def get_latest_audio(self, num_samples, current_frame_idx, fps=25):
        """Calculates audio chunk including the END of the current frame."""
        # Add +1 to current_frame_idx to get the end time of the frame
        end_sample = int(((current_frame_idx + 1) / fps) * self.sample_rate)
        start_sample = max(0, end_sample - num_samples)
        
        chunk = self.audio_data[start_sample:end_sample]
        
        if len(chunk) < num_samples:
            pad = np.zeros(num_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate((pad, chunk))
            
        return chunk
        
    def close(self):
        pass # Nothing to close for local array