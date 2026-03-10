# eval_merged.py
# MEASURING STOP DETERMINISM (BASELINE VS PROPOSED)
# -----------------------------------------------

import re
import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import starmap, repeat

import cv2 as cv
import numpy as np
from _mediapipePoseEstimation import HandDetector
from hgr_box_gate import BoxGate 

# [GLOBALS]__________________________________________________________
try:
    cfg = json.load(open("./allConfigs.jsonc"))
except:
    print("WARNING: Config file not found, using defaults.")
    cfg = {"MAX_HISTORY": 150, "HISTORY_BUFFER": 20, "LOGGER_THRESHOLD": 0.2, 
           "hgr_log": True, "FRAME_SIZE": 640, "VERBOSE": False}

# --- SAVE DIRECTORIES ---
BASE_SAVE_DIR = "./metric_tests/baseline_manual"
PROP_SAVE_DIR = "./metric_tests/proposed_auto"
Path(BASE_SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(PROP_SAVE_DIR).mkdir(parents=True, exist_ok=True)
# ---------------------------------------------

# --- INDEPENDENT BUFFERS ---
# Baseline needs a rolling buffer
base_deque = deque(maxlen=cfg["MAX_HISTORY"])
base_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
base_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])

# Proposed needs BoxGate controlled buffers
prop_deque = deque(maxlen=cfg["MAX_HISTORY"])
pre_buffer = deque(maxlen=6) 

# [FUNCTIONS]________________________________________________________
def gs_logger_baseline():
    global base_deque
    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    print(f"\n[BASELINE] Manual-Saving sample: {gs_tag}")

    save_path = Path(f"{BASE_SAVE_DIR}/{gs_tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    n_skeletons = min(len(base_deque), base_length)
    gs = np.array(list(starmap(base_deque.popleft, repeat((), n_skeletons))))

    np.save(f"{save_path}/gs_sequence", gs)
    print(f" -> Saved {len(gs)} frames to {save_path}")

def gs_logger_proposed():
    global prop_deque
    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    print(f"\n[PROPOSED] Auto-Saving sample: {gs_tag}")

    save_path = Path(f"{PROP_SAVE_DIR}/{gs_tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    gs = np.array(list(prop_deque)) # Save exactly what the Gate captured

    np.save(f"{save_path}/gs_sequence", gs)
    print(f" -> Saved {len(gs)} frames to {save_path}")

def live_stream_merged(nD):
    time.sleep(1)
    nD = nD.upper()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "METRIC 1: MERGED EVALUATION"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, x=50, y=15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()
    
    # Counters
    base_samples = 0
    prop_samples = 0
    TARGET_SAMPLES = 10
    
    print(f"INFO: Ready. Collecting {TARGET_SAMPLES} samples for BOTH methods...")

    while True:
        success, img = cap.read()
        if not success: continue
        
        img = cv.flip(img, 1)
        hand_data, img = detector.findHands(img, draw=True)
        
        prop_state = "NO_HAND"

        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]
            
            # 1. BASELINE CONTINUOUS TRACKING
            if base_samples < TARGET_SAMPLES:
                base_deque.append(lmCoords)

            # 2. PROPOSED BOXGATE TRACKING
            if prop_samples < TARGET_SAMPLES:
                prop_state, val = gate.process(lmCoords)
                pre_buffer.append(lmCoords)

                if prop_state == "RECORDING":
                    if gate.frame_count == 1:
                        prop_deque.clear()
                        prop_deque.extend(pre_buffer) 
                        print(">>> PROPOSED: STARTED")
                    
                    prop_deque.append(lmCoords)
                    cv.putText(img, "PROPOSED: RECORDING", (10, 60), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                elif prop_state == "FINISHED":
                    prop_samples += 1
                    print(f" -> Proposed Progress: {prop_samples}/{TARGET_SAMPLES}")
                    
                    cv.putText(img, "PROPOSED: FINISHED - SAVING", (10, 60), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv.imshow(window_name, img)
                    cv.waitKey(50) 
                    
                    gs_logger_proposed()
                    
                    prop_deque.clear()
                    pre_buffer.clear()
                    gate.reset()

                elif prop_state == "IDLE":
                    if gate.anchor_pos is not None:
                        wrist_raw = hand["lmCoords_2D"][0]
                        cx, cy = int(wrist_raw[0]), int(wrist_raw[1])
                        cv.circle(img, (cx, cy), 50, (0,255,0), 1)
                    cv.putText(img, "PROPOSED: IDLE", (10, 60), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            gate.reset()
            pre_buffer.clear()
            cv.putText(img, "WAITING FOR HAND", (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # UI Overlays for Progress
        cv.putText(img, f"Baseline (Spacebar): {base_samples}/{TARGET_SAMPLES}", (10, 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if base_samples < TARGET_SAMPLES else (0, 255, 0), 2)
        cv.putText(img, f"Proposed (Auto): {prop_samples}/{TARGET_SAMPLES}", (350, 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if prop_samples < TARGET_SAMPLES else (0, 255, 0), 2)

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        # --- KEYBOARD CONTROLS ---
        if key == 27: # ESC
            break
            
        elif key == ord("c"):
            base_deque.clear()
            prop_deque.clear()
            gate.reset()

        # --- MANUAL TRIGGER (BASELINE) ---
        elif (key == 32) and (len(base_deque) > base_minimum) and (base_samples < TARGET_SAMPLES):
            gs_logger_baseline()
            base_samples += 1
            print(f" -> Baseline Progress: {base_samples}/{TARGET_SAMPLES}")
            
            # Visual flash for manual capture
            cv.rectangle(img, (0,0), (640,480), (0,0,255), -1)
            cv.imshow(window_name, img)
            cv.waitKey(200)

        # --- GLOBAL EXIT CONDITION ---
        if base_samples >= TARGET_SAMPLES and prop_samples >= TARGET_SAMPLES:
            print("\n>>> ALL COLLECTIONS COMPLETE (10/10) <<<")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    live_stream_merged(nD="3d")