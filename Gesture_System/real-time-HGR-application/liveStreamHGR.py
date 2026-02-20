# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# pyright: reportOptionalMemberAccess=false
# -----------------------------------------------
import sys
import re
import json
import time
import subprocess
from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import starmap, repeat

import cv2 as cv
import numpy as np
import matplotlib.colors as mc
from _mediapipePoseEstimation import HandDetector
from _helperFunctions import hgrLogger

# Import our Geometric Gate
from hgr_box_gate import BoxGate 


# [GLOBALS]__________________________________________________________
cfg = json.load(open("./allConfigs.jsonc"))
# ---
gs_deque = deque(maxlen=cfg["MAX_HISTORY"])
gs_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
gs_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])
hgr_log = cfg["hgr_log"]
# ---
str_colors = dict(cfg["mp_fingers_colors"])
connection_map = cfg["mp_connection_map"]
finger_tips = cfg["mp_finger_tips"]
mp_drawings = cfg["MP_DRAWINGS"]

rgb_colors = []

# [FUNCTIONS]________________________________________________________
def _color_fingers():
    global rgb_colors

    nodes = str_colors.keys()
    rgb_colors = [mc.to_rgb(mc.CSS4_COLORS[str_colors[n]])[::-1] for n in nodes]
    rgb_colors = (np.array(rgb_colors) * 255).astype(int).tolist()


def _draw_landmarks(img, lmCoords_2D):
    lmCoords_2D = lmCoords_2D[:, :-1].astype(int)

    for (node1, node2) in connection_map:
        cv.line(img, lmCoords_2D[node1], lmCoords_2D[node2], rgb_colors[node1], 2)

    for (node, coords) in enumerate(lmCoords_2D):
        if node in finger_tips:
            cv.circle(img, coords, 5, rgb_colors[node], cv.FILLED)
            cv.circle(img, coords, 3, (0, 0, 0), cv.FILLED)
        else:
            cv.circle(img, coords, 3, rgb_colors[node], cv.FILLED)

    return img


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
    time.sleep(5)
    nD = nD.upper()
    assert nD in ["2D", "3D"], "ValueError@ nD parameter"

    # --- init camera window, set frame dimensions, and keep on top
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "e2eET HGR: Mediapipe Skeleton Estimation"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    cv.moveWindow(window_name, x=50, y=15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()
    capture_count = 0
    
    pre_buffer = deque(maxlen=6) # Pre-Trigger Buffer

    # --- [NEW] VISUALIZATION MEMORY ---
    # Blueprints from test_full_viz.py to implement the drawing hack.
    frozen_pos_2d = None     # The Static 2D Geofence (pixels)
    frozen_scale_px = None   # The Static 2D Aura (pixels)
    prev_cx, prev_cy = 0, 0  # Historical pixel coordinates
    # ----------------------------------
    
    _color_fingers()
    print("INFO: Initialized <liveStreamHGR.py> (Position + Scale Viz) ...")

    while True:
        success, img = cap.read()
        if not success: continue 
        
        hand_data, img = detector.findHands(img, draw=mp_drawings)
        h, w, c = img.shape
        img = cv.flip(img, 1) # Flip Image
        
        state = "NO_HAND"

        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]
            
            # Mirror Fix for drawing
            wrist_raw = hand["lmCoords_2D"][0]
            cx, cy = w - int(wrist_raw[0]), int(wrist_raw[1])

            # --- [VISUALIZATION PREP] ---
            # Calculate current pixel hand aura ("Size Balloon")
            tips = lmCoords[[4,8,12,16,20]]
            curr_scale_3d = np.mean(np.linalg.norm(tips - lmCoords[0], axis=1))
            curr_radius_px = int(curr_scale_3d * (60 / 0.08)) # Drawing ratio

            # --- [GATE LOGIC] ---
            state, val = gate.process(lmCoords)

            # always update pre-buffer
            pre_buffer.append(lmCoords)

            # --- [RECORDING + VISUAL ANCHORING LOGIC] ---
            if state == "RECORDING":
                if gate.frame_count == 1:
                    # PRE-BUFFER INJECTION
                    gs_deque.clear() 
                    gs_deque.extend(pre_buffer) 
                    print(f"\n>>> STARTED (Injected {len(pre_buffer)} frames history)")

                    # --- [NEW] VISUAL ANCHOR FREEZE ---
                    # We freeze the concentric "Ghosts" using historical pixel data.
                    frozen_pos_2d = (prev_cx, prev_cy)
                    frozen_scale_px = curr_radius_px
                
                # Continue appending normal frames
                gs_deque.append(lmCoords)
                print(f"REC: {len(gs_deque)}", end='\r') if cfg["VERBOSE"] else None

            # FINISHED
            elif state == "FINISHED":
                capture_count += 1
                print(f"\n>>> CAPTURE {capture_count} SUCCESS! Saving {len(gs_deque)} frames...")
                gs_logger() 
                gs_deque.clear()
                pre_buffer.clear() 

            # REJECTED
            elif state == "REJECTED":
                gs_deque.clear()
            
            # --- [FULL SENSORY VISUALIZATION (test_full_viz.py)] ---
            if gate.anchor_pos is not None:
                
                # [A] IDLE / WARMUP (Magnet Mode - Concentric concentric rings following wrist)
                if state == "IDLE" or state == "WARMUP":
                    frozen_pos_2d = None 
                    frozen_scale_px = None
                    
                    color = (0, 255, 0) # Green
                    if state == "WARMUP": color = (0, 255, 255) # Yellow
                    
                    # 1. Draw Position Geofence (Safe Zone - Static Center)
                    # (Follows wrist in Idle)
                    cv.circle(img, (cx, cy), 60, (255, 255, 255), 1) 
                    
                    # 2. Draw Hand Size Aura ( balloon inflated - concentric)
                    cv.circle(img, (cx, cy), curr_radius_px, color, 2)
                    
                    # 3. Status Wrist Dot
                    cv.circle(img, (cx, cy), 5, color, -1)
                    
                    label = state if state == "WARMUP" else "SAFE ZONE"
                    cv.putText(img, label, (cx-60, cy-70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # [B] RECORDING / FINISHED (Anchor Mode - Ghosts frozen, Real concentric concentric moving)
                elif state == "RECORDING" or state == "FINISHED":
                    # Frozen Ghosts were set at frame_count == 1
                    ax, ay = frozen_pos_2d
                    
                    # Color Logic
                    active_color = (0, 0, 255) # Red
                    if state == "FINISHED": active_color = (255, 0, 0) # Blue
                    
                    # 1. DRAW CONCENTRIC GHOSTS (Reference / Background)
                    # Ghost Position Box (Ghost Center)
                    cv.circle(img, (ax, ay), 60, (150, 150, 150), 2) 
                    # Ghost Size Aura (balloon reference - concentric)
                    cv.circle(img, (cx, cy), frozen_scale_px, (200, 200, 200), 1, cv.LINE_AA)
                    
                    # 2. DRAW CONCENTRIC REALITY (Active / Foreground)
                    # Real Position (Active Wrist Dot)
                    cv.circle(img, (cx, cy), 8, active_color, -1)
                    # Real Size Aura (balloon active - concentric)
                    cv.circle(img, (cx, cy), curr_radius_px, active_color, 3)
                    
                    # 3. DRAW LEASH
                    dist_px = np.linalg.norm(np.array([cx, cy]) - np.array([ax, ay]))
                    cv.line(img, (ax, ay), (cx, cy), active_color, 2)
                    
                    # 4. TRIGGER LABEL (Swipe vs. Grab)
                    # If leash is short, it's likely a grab. If long, it's a swipe.
                    mode_text = "SWIPE!" if dist_px > 50 else "GRAB/SIZE!"
                    cv.putText(img, mode_text, (cx-40, cy-80), cv.FONT_HERSHEY_SIMPLEX, 0.6, active_color, 2)

            # Update Viz History for next frame
            prev_cx, prev_cy = cx, cy

        else:
            gate.reset()
            pre_buffer.clear() 
            frozen_pos_2d = None # Reset visual anchors

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        if key == 27:  # escape key
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