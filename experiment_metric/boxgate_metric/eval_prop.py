# metric1_proposed.py
# MEASURING STOP DETERMINISM (PROPOSED)
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

# --- OVERRIDE SAVE PATH FOR METRIC TESTING ---
TEST_SAVE_DIR = "./metric_tests/proposed_auto"
Path(TEST_SAVE_DIR).mkdir(parents=True, exist_ok=True)
# ---------------------------------------------

gs_deque = deque(maxlen=cfg["MAX_HISTORY"])
pre_buffer = deque(maxlen=6) 

# [FUNCTIONS]________________________________________________________
def gs_logger():
    global gs_deque
    
    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    print(f"\n[PROPOSED] Auto-Saving sample: {gs_tag}")

    save_path = Path(f"{TEST_SAVE_DIR}/{gs_tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    gs = np.array(list(gs_deque)) # Save exactly what the Gate captured

    np.save(f"{save_path}/gs_sequence", gs)
    print(f" -> Saved {len(gs)} frames to {save_path}")

def live_stream_hgr(nD):
    time.sleep(1)
    nD = nD.upper()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "METRIC 1: PROPOSED (BOXGATE STOP)"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, x=50, y=15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()
    
    # [NEW] Counter to stop after 10 samples
    sample_count = 0 
    TARGET_SAMPLES = 10
    
    print(f"INFO: Ready. Collecting {TARGET_SAMPLES} samples...")

    while True:
        success, img = cap.read()
        if not success: continue
        
        # [FIX] Flip early
        img = cv.flip(img, 1)

        hand_data, img = detector.findHands(img, draw=True)
        
        state = "NO_HAND"

        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]
            
            state, val = gate.process(lmCoords)
            pre_buffer.append(lmCoords)

            if state == "RECORDING":
                if gate.frame_count == 1:
                    gs_deque.clear()
                    gs_deque.extend(pre_buffer) 
                    print(">>> STARTED")
                
                gs_deque.append(lmCoords)
                cv.putText(img, f"RECORDING | Samples: {sample_count}/{TARGET_SAMPLES}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            elif state == "FINISHED":
                # [NEW] Increment Counter
                sample_count += 1
                print(f" -> Progress: {sample_count}/{TARGET_SAMPLES}")

                cv.putText(img, "FINISHED - SAVING", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv.imshow(window_name, img)
                cv.waitKey(50) 
                
                gs_logger()
                
                gs_deque.clear()
                pre_buffer.clear()
                gate.reset()
                
                # [NEW] Check Exit Condition
                if sample_count >= TARGET_SAMPLES:
                    print("\n>>> COLLECTION COMPLETE (10/10) <<<")
                    break

            elif state == "IDLE":
                if gate.anchor_pos is not None:
                    wrist_raw = hand["lmCoords_2D"][0]
                    cx, cy = int(wrist_raw[0]), int(wrist_raw[1])
                    cv.circle(img, (cx, cy), 50, (0,255,0), 1)
                
                cv.putText(img, f"IDLE | Samples: {sample_count}/{TARGET_SAMPLES}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            gate.reset()
            pre_buffer.clear()
            cv.putText(img, f"WAITING | Samples: {sample_count}/{TARGET_SAMPLES}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        if key == 27: break
        elif key == ord("c"):
            gs_deque.clear()
            gate.reset()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    live_stream_hgr(nD="3d")