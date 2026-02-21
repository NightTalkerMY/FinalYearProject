# metric1_baseline.py
# MEASURING STOP DETERMINISM (BASELINE)
# -----------------------------------------------

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
# from _helperFunctions import hgrLogger

# [GLOBALS]__________________________________________________________
try:
    cfg = json.load(open("./allConfigs.jsonc"))
except:
    print("WARNING: Config file not found, using defaults.")
    cfg = {"MAX_HISTORY": 150, "HISTORY_BUFFER": 20, "LOGGER_THRESHOLD": 0.2, 
           "hgr_log": True, "FRAME_SIZE": 640, "VERBOSE": False}

# --- OVERRIDE SAVE PATH FOR METRIC TESTING ---
TEST_SAVE_DIR = "./metric_tests/baseline_manual"
Path(TEST_SAVE_DIR).mkdir(parents=True, exist_ok=True)
# ---------------------------------------------

gs_deque = deque(maxlen=cfg["MAX_HISTORY"])
gs_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
gs_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])
hgr_log = cfg["hgr_log"]

# Dummy colors if config fails
str_colors = {"thumb": "red", "index": "green"} 
connection_map = []
finger_tips = [4, 8, 12, 16, 20]
mp_drawings = True

# [FUNCTIONS]________________________________________________________
def gs_logger():
    global gs_deque
    
    # Create timestamp
    gs_tag = re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")
    print(f"\n[BASELINE] Saving sample: {gs_tag}")

    # Create folder for this specific sample inside the test directory
    save_path = Path(f"{TEST_SAVE_DIR}/{gs_tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    n_skeletons = min(len(gs_deque), gs_length)
    gs = np.array(list(starmap(gs_deque.popleft, repeat((), n_skeletons))))

    np.save(f"{save_path}/gs_sequence", gs)
    print(f" -> Saved {len(gs)} frames to {save_path}")

def live_stream_hgr(nD):
    time.sleep(1)
    nD = nD.upper()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "METRIC 1: BASELINE (MANUAL STOP)"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, x=50, y=15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    
    # [NEW] Counter to stop after 10 samples
    sample_count = 0 
    TARGET_SAMPLES = 10

    print(f"INFO: Ready. Collecting {TARGET_SAMPLES} samples...")

    while True:
        success, img = cap.read()
        if not success: continue

        # [FIX] Flip early
        img = cv.flip(img, 1)

        hand, img = detector.findHands(img, draw=mp_drawings)

        if hand:
            hand_data = hand[0]
            lmCoords = hand_data[f"lmCoords_{nD}"]
            gs_deque.append(lmCoords)

            # Visual feedback
            cv.putText(img, f"REC (Buffer Filling) | Samples: {sample_count}/{TARGET_SAMPLES}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv.putText(img, f"WAITING FOR HAND | Samples: {sample_count}/{TARGET_SAMPLES}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        if key == 27: break

        # --- MANUAL TRIGGER ---
        elif (key == 32) and (len(gs_deque) > gs_minimum):
            gs_logger()
            
            # [NEW] Increment and Check
            sample_count += 1
            print(f" -> Progress: {sample_count}/{TARGET_SAMPLES}")
            
            # Visual flash
            cv.rectangle(img, (0,0), (640,480), (0,255,0), -1)
            cv.imshow(window_name, img)
            cv.waitKey(200)

            if sample_count >= TARGET_SAMPLES:
                print("\n>>> COLLECTION COMPLETE (10/10) <<<")
                break

        elif key == ord("c"):
            gs_deque.clear()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    live_stream_hgr(nD="3d")