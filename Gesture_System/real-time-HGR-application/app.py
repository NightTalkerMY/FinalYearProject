# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: WebRTC Receiver
# -----------------------------------------------
import sys
import re
import json
import time
import subprocess
import threading
import queue
import asyncio
import aiohttp
import av
from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import starmap, repeat

import cv2 as cv
import numpy as np
import matplotlib.colors as mc
from aiortc import RTCPeerConnection, RTCSessionDescription

# --- IMPORT YOUR MODULES ---
from _mediapipePoseEstimation import HandDetector
from _helperFunctions import hgrLogger
from hgr_box_gate import BoxGate 

# [CONFIGURATION]____________________________________________________
# REPLACE THIS WITH YOUR MEDIAMTX SERVER IP
# Note the endpoint ends in '/whep' for subscribers (WHIP is for publishers)
SERVER_URL = "http://127.0.0.1:8889/cam1/whep" 

cfg = json.load(open("./allConfigs.jsonc"))

# [GLOBALS]__________________________________________________________
gs_deque = deque(maxlen=cfg["MAX_HISTORY"])
gs_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
gs_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])
hgr_log = cfg["hgr_log"]

str_colors = dict(cfg["mp_fingers_colors"])
connection_map = cfg["mp_connection_map"]
finger_tips = cfg["mp_finger_tips"]
mp_drawings = cfg["MP_DRAWINGS"]
rgb_colors = []

# [HELPER CLASS: WEBRTC RECEIVER]____________________________________
# [HELPER CLASS: WEBRTC RECEIVER]____________________________________
class WebRTCVideoCapture:
    """
    A drop-in replacement for cv.VideoCapture that reads from a WebRTC (WHEP) stream.
    """
    def __init__(self, url):
        self.url = url
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.pc = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print(f"[WebRTC] Connecting to {self.url}...")
        # Give it a moment to connect
        time.sleep(2) 

    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        else:
            return False, None

    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _run_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consume_stream())

    async def _consume_stream(self):
        self.pc = RTCPeerConnection()

        # --- IMPORTANT FIX: Request Video Recv ---
        self.pc.addTransceiver("video", direction="recvonly")
        # -----------------------------------------

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print("[WebRTC] Video track received!")
                asyncio.ensure_future(self._read_track(track))

        # 1. Create Offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # 2. POST to MediaMTX (WHEP)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.url,
                    data=self.pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"}
                ) as resp:
                    if resp.status != 201:
                        print(f"[WebRTC] Error: {resp.status} - {await resp.text()}")
                        return
                    answer = await resp.text()
            except Exception as e:
                print(f"[WebRTC] Connection Failed: {e}")
                return

        # 3. Set Remote Description
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer, type="answer")
        )

        # Keep alive
        while self.running:
            await asyncio.sleep(1)
        
        await self.pc.close()

    async def _read_track(self, track):
        while self.running:
            try:
                frame = await track.recv()
                # Convert to numpy (RGB -> BGR for OpenCV)
                img = frame.to_ndarray(format="bgr24")
                
                # Manage queue size to keep latency low
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(img)
            except Exception:
                pass

# [FUNCTIONS]________________________________________________________
def _color_fingers():
    global rgb_colors
    nodes = str_colors.keys()
    rgb_colors = [mc.to_rgb(mc.CSS4_COLORS[str_colors[n]])[::-1] for n in nodes]
    rgb_colors = (np.array(rgb_colors) * 255).astype(int).tolist()

def gs_logger():
    global gs_deque
    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    hgrLogger(f"{'-'*25}\n>HGR: @{gs_tag}: {len(gs_deque)=:02}->>-", log=hgr_log, end="")

    gs_tag = Path(f"{cfg['data_directory']}/{gs_tag}")
    gs_tag.mkdir(parents=True, exist_ok=True)

    n_skeletons = min(len(gs_deque), gs_length)
    gs = np.array(list(starmap(gs_deque.popleft, repeat((), n_skeletons))))

    np.save(f"{gs_tag}/gs_sequence", gs)
    hgrLogger(f"{len(gs_deque):02} | {gs.shape=}", log=hgr_log)


def live_stream_hgr(nD):
    time.sleep(1) # Reduced sleep
    nD = nD.upper()
    
    # --- WEBRTC CAPTURE REPLACEMENT ---
    # Instead of cv.VideoCapture(0), we use our custom class
    cap = WebRTCVideoCapture(SERVER_URL)
    cap.start()
    
    window_name = "e2eET HGR: WebRTC Stream"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    cv.moveWindow(window_name, x=50, y=15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()
    capture_count = 0
    
    pre_buffer = deque(maxlen=6)

    # --- VISUALIZATION MEMORY ---
    frozen_pos_2d = None
    frozen_scale_px = None
    prev_cx, prev_cy = 0, 0
    
    _color_fingers()
    print("INFO: Initialized <liveStreamHGR.py> (WebRTC Mode)...")

    while True:
        # This now reads from the WebRTC queue
        success, img = cap.read()
        
        if not success:
            time.sleep(0.01) # Don't burn CPU if waiting for frames
            continue 
        
        hand_data, img = detector.findHands(img, draw=mp_drawings)
        h, w, c = img.shape
        img = cv.flip(img, 1)
        
        state = "NO_HAND"

        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]
            
            wrist_raw = hand["lmCoords_2D"][0]
            cx, cy = w - int(wrist_raw[0]), int(wrist_raw[1])

            # --- [VISUALIZATION PREP] ---
            tips = lmCoords[[4,8,12,16,20]]
            curr_scale_3d = np.mean(np.linalg.norm(tips - lmCoords[0], axis=1))
            curr_radius_px = int(curr_scale_3d * (60 / 0.08))

            # --- [GATE LOGIC] ---
            state, val = gate.process(lmCoords)
            pre_buffer.append(lmCoords)

            # --- [RECORDING LOGIC] ---
            if state == "RECORDING":
                if gate.frame_count == 1:
                    gs_deque.clear() 
                    gs_deque.extend(pre_buffer) 
                    print(f"\n>>> STARTED (Injected {len(pre_buffer)} frames)")
                    frozen_pos_2d = (prev_cx, prev_cy)
                    frozen_scale_px = curr_radius_px
                
                gs_deque.append(lmCoords)
                print(f"REC: {len(gs_deque)}", end='\r') if cfg["VERBOSE"] else None

            elif state == "FINISHED":
                capture_count += 1
                print(f"\n>>> CAPTURE {capture_count} SUCCESS!")
                gs_logger() 
                gs_deque.clear()
                pre_buffer.clear() 

            elif state == "REJECTED":
                gs_deque.clear()
            
            # --- [VISUALIZATION DRAWING] ---
            if gate.anchor_pos is not None:
                if state == "IDLE" or state == "WARMUP":
                    frozen_pos_2d = None 
                    frozen_scale_px = None
                    color = (0, 255, 0) if state == "IDLE" else (0, 255, 255)
                    
                    cv.circle(img, (cx, cy), 60, (255, 255, 255), 1) 
                    cv.circle(img, (cx, cy), curr_radius_px, color, 2)
                    cv.circle(img, (cx, cy), 5, color, -1)
                    label = state if state == "WARMUP" else "SAFE ZONE"
                    cv.putText(img, label, (cx-60, cy-70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                elif state == "RECORDING" or state == "FINISHED":
                    if frozen_pos_2d:
                        ax, ay = frozen_pos_2d
                        active_color = (0, 0, 255) if state == "RECORDING" else (255, 0, 0)
                        
                        cv.circle(img, (ax, ay), 60, (150, 150, 150), 2) 
                        if frozen_scale_px:
                            cv.circle(img, (cx, cy), frozen_scale_px, (200, 200, 200), 1, cv.LINE_AA)
                        
                        cv.circle(img, (cx, cy), 8, active_color, -1)
                        cv.circle(img, (cx, cy), curr_radius_px, active_color, 3)
                        
                        cv.line(img, (ax, ay), (cx, cy), active_color, 2)
                        dist_px = np.linalg.norm(np.array([cx, cy]) - np.array([ax, ay]))
                        mode_text = "SWIPE!" if dist_px > 50 else "GRAB/SIZE!"
                        cv.putText(img, mode_text, (cx-40, cy-80), cv.FONT_HERSHEY_SIMPLEX, 0.6, active_color, 2)

            prev_cx, prev_cy = cx, cy

        else:
            gate.reset()
            pre_buffer.clear() 
            frozen_pos_2d = None

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        if key == 27: 
            cap.release()
            break
        elif key == ord("c"):
            gs_deque.clear()
            gate.reset()

if __name__ == "__main__":
    python_exe = sys.executable  # this is your venv's python

    gesture_inference = subprocess.Popen([python_exe, "gestureClassInference.py"])
    data_level_fusion = subprocess.Popen([python_exe, "dataLevelFusion.py"])
    vispy_output_gui  = subprocess.Popen([python_exe, "vispyOutputGUI.py"])

    try:
        live_stream_hgr(nD="3d")
        print("\nINFO: Exiting liveStreamHGR normally. ", end="")

    except Exception as ex: print(f"\nERROR: >- {ex}. -< ", end="")

    finally:
        gesture_inference.terminate()
        data_level_fusion.terminate()
        vispy_output_gui.terminate()
        cv.destroyAllWindows()
        print("All child processes terminated!")