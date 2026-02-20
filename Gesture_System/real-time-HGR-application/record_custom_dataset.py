# # record_custom_dataset.py
# # ---------------------------------------------------------
# import sys
# import time
# import json
# import argparse
# import subprocess
# import numpy as np
# import cv2 as cv
# from pathlib import Path
# from collections import deque

# # Import your helper classes
# from _mediapipePoseEstimation import HandDetector

# # Load Configs to find the correct folders
# try:
#     cfg = json.load(open("./allConfigs.jsonc"))
# except FileNotFoundError:
#     print("ERROR: Could not find 'allConfigs.jsonc'. Make sure you are in the experiments.server folder.")
#     sys.exit(1)

# # Constants
# DATA_DIR = Path(cfg["data_directory"])      # Where we send raw data (.npy)
# IMAGES_DIR = Path(cfg["images_directory"])  # Where the final images end up
# MAX_HISTORY = cfg["MAX_HISTORY"]

# def main():
#     # --- 1. Setup Arguments & Constants ---
#     parser = argparse.ArgumentParser(description="Record custom dataset for HGR")
#     parser.add_argument("--label", type=str, required=True, help="Name of the gesture (e.g., SwipeLeft)")
#     args = parser.parse_args()
    
#     # We still keep the phases so you know what to record, 
#     # but we won't block you if the detector gets it wrong.
#     # Adjust CLOSE_THRESHOLD based on your camera resolution.
#     CLOSE_THRESHOLD = 28000 
#     PHASES = [
#         {"name": "RIGHT HAND (Far)",   "target_dist": "Far"},   # Samples 1-25
#         {"name": "RIGHT HAND (Close)", "target_dist": "Close"}, # Samples 26-50
#         {"name": "LEFT HAND (Far)",    "target_dist": "Far"},   # Samples 51-75
#         {"name": "LEFT HAND (Close)",  "target_dist": "Close"}  # Samples 76-100
#     ]
#     SAMPLES_PER_PHASE = 25

#     # --- 2. Launch Renderer ---
#     print(f">> Launching Vispy Renderer (dataLevelFusion.py)...")
#     fusion_process = subprocess.Popen(args="python ./dataLevelFusion.py", shell=True)
#     time.sleep(3) 

#     # --- 3. Setup Camera & Detector ---
#     cap = cv.VideoCapture(0, cv.CAP_DSHOW)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))
    
#     detector = HandDetector(detectionCon=0.85, maxHands=1)
    
#     gs_deque = deque(maxlen=MAX_HISTORY)
#     is_recording = False
    
#     # Initialize Sample Count
#     existing_samples = list(IMAGES_DIR.glob(f"{args.label}_*"))
#     total_sample_count = len(existing_samples) + 1

#     print(f"\n{'='*50}")
#     print(f" RECORDING SESSION: {args.label}")
#     print(f" [SPACE] to Start/Stop | [ESC] to Quit")
#     print(f"{'='*50}\n")

#     try:
#         while True:
#             success, img = cap.read()
#             if not success: break
            
#             img = cv.flip(img, 1)
            
#             # --- Determine Current Phase (For Display Only) ---
#             current_phase_idx = min((total_sample_count - 1) // SAMPLES_PER_PHASE, 3)
#             current_phase = PHASES[current_phase_idx]
            
#             # Progress (e.g., 5/25)
#             phase_progress = (total_sample_count - 1) % SAMPLES_PER_PHASE + 1
#             if total_sample_count > 100: phase_progress = SAMPLES_PER_PHASE

#             # --- Hand Detection ---
#             hand, img = detector.findHands(img, draw=True)
#             display_img = img.copy()

#             # --- UI: Header Info ---
#             cv.rectangle(display_img, (0,0), (img.shape[1], 60), (0,0,0), -1)
#             cv.putText(display_img, f"PHASE {current_phase_idx+1}: {current_phase['name']}", (10, 25), 
#                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             cv.putText(display_img, f"Count: {phase_progress}/{SAMPLES_PER_PHASE}", (10, 50), 
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

#             if hand:
#                 hand_data = hand[0]
#                 lmCoords = hand_data["lmCoords_3D"]
#                 bbox = hand_data.get("bbox", [0,0,0,0]) 
                
#                 # --- Calculate Distance (Area) ---
#                 area = bbox[2] * bbox[3]
#                 dist_status = "Close" if area > CLOSE_THRESHOLD else "Far"
                
#                 # Check if distance matches target (Visual Feedback only)
#                 is_dist_correct = (dist_status == current_phase["target_dist"])
                
#                 # Color: Green if distance is correct, Yellow if not (Warning, but not blocking)
#                 status_color = (0, 255, 0) if is_dist_correct else (0, 255, 255)
                
#                 # Draw Box
#                 x, y, w, h = bbox
#                 cv.rectangle(display_img, (x, y), (x+w, y+h), status_color, 2)
                
#                 # Display Status
#                 status_text = f"Dist: {dist_status} ({area})"
#                 cv.putText(display_img, status_text, (x, y - 10), 
#                            cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

#                 # Distance Warning (Optional help)
#                 if not is_dist_correct:
#                     target = current_phase['target_dist']
#                     msg = "MOVE CLOSER" if target == "Close" else "MOVE BACK"
#                     cv.putText(display_img, f"{msg}", (10, 100), 
#                                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # --- Recording Logic ---
#                 if is_recording:
#                     gs_deque.append(lmCoords)
#                     # Visual: Red Recording Indicator
#                     cv.circle(display_img, (img.shape[1]-50, 40), 20, (0, 0, 255), -1)
#                     cv.putText(display_img, "REC", (img.shape[1]-65, 45), 
#                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             else:
#                 if is_recording:
#                     cv.putText(display_img, "HAND LOST!", (10, 200), 
#                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#             cv.imshow("Custom Dataset Recorder", display_img)
#             key = cv.waitKey(1)

#             # --- Controls ---
#             if key == 27: # ESC
#                 break
#             elif key == 32: # SPACE
#                 if not is_recording:
#                     is_recording = True
#                     gs_deque.clear()
#                 else:
#                     is_recording = False
#                     if len(gs_deque) > 15:
#                         folder_name = f"{args.label}_{total_sample_count:03d}"
#                         save_path = DATA_DIR / folder_name
#                         save_path.mkdir(parents=True, exist_ok=True)
#                         np.save(f"{save_path}/gs_sequence.npy", np.array(gs_deque))
                        
#                         print(f">> Saved #{total_sample_count} ({current_phase['name']})")
#                         total_sample_count += 1
#                     else:
#                         print(">> Too short! Discarded.")

#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         print("\n>> Closing Renderer and Exiting...")
#         fusion_process.terminate()
#         cap.release()
#         cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# record_custom_dataset.py
# ---------------------------------------------------------
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import cv2 as cv
from pathlib import Path
from collections import deque

# Import your helper classes
from _mediapipePoseEstimation import HandDetector

# Load Configs to find the correct folders
try:
    cfg = json.load(open("./allConfigs.jsonc"))
except FileNotFoundError:
    print("ERROR: Could not find 'allConfigs.jsonc'. Make sure you are in the experiments.server folder.")
    sys.exit(1)

# Constants
DATA_DIR = Path(cfg["data_directory"])      # Where we send raw data (.npy)
IMAGES_DIR = Path(cfg["images_directory"])  # Where the final images end up
MAX_HISTORY = cfg["MAX_HISTORY"]

def main():
    # --- 1. Setup Arguments & Constants ---
    parser = argparse.ArgumentParser(description="Record custom dataset for HGR")
    parser.add_argument("--label", type=str, required=True, help="Name of the gesture (e.g., SwipeLeft)")
    args = parser.parse_args()
    
    # Adjust CLOSE_THRESHOLD based on your camera resolution.
    CLOSE_THRESHOLD = 28000 
    PHASES = [
        {"name": "RIGHT HAND (Far)",   "target_dist": "Far"},   # Samples 1-25
        {"name": "RIGHT HAND (Close)", "target_dist": "Close"}, # Samples 26-50
        {"name": "LEFT HAND (Far)",    "target_dist": "Far"},   # Samples 51-75
        {"name": "LEFT HAND (Close)",  "target_dist": "Close"}  # Samples 76-100
    ]
    SAMPLES_PER_PHASE = 25

    # --- 2. Launch Renderer ---
    print(f">> Launching Vispy Renderer (dataLevelFusion.py)...")
    fusion_process = subprocess.Popen(args="python ./dataLevelFusion.py", shell=True)
    time.sleep(3) 

    # --- 3. Setup Camera & Detector ---
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, (cfg["FRAME_SIZE"] * 9 / 16))
    
    detector = HandDetector(detectionCon=0.85, maxHands=1)
    
    gs_deque = deque(maxlen=MAX_HISTORY)
    is_recording = False
    
    # Initialize Sample Count
    existing_samples = list(IMAGES_DIR.glob(f"{args.label}_*"))
    total_sample_count = len(existing_samples) + 1

    print(f"\n{'='*50}")
    print(f" RECORDING SESSION: {args.label}")
    print(f" [SPACE] to Start/Stop | [ESC] to Quit")
    print(f"{'='*50}\n")

    try:
        while True:
            success, img = cap.read()
            if not success: break
            
            # [FIX] Do NOT flip 'img' yet. We detect on the RAW image
            # to ensure the coordinates match liveStreamHGR.py logic.
            img_h, img_w, _ = img.shape
            
            # --- Determine Current Phase (For Display Only) ---
            current_phase_idx = min((total_sample_count - 1) // SAMPLES_PER_PHASE, 3)
            current_phase = PHASES[current_phase_idx]
            
            # Progress (e.g., 5/25)
            phase_progress = (total_sample_count - 1) % SAMPLES_PER_PHASE + 1
            if total_sample_count > 100: phase_progress = SAMPLES_PER_PHASE

            # --- Hand Detection (ON RAW IMAGE) ---
            hand, img = detector.findHands(img, draw=True)
            
            # --- Prepare Display Image (NOW WE FLIP) ---
            # Landmarks drawn by detector will flip correctly with the image
            display_img = cv.flip(img, 1)

            # --- UI: Header Info ---
            cv.rectangle(display_img, (0,0), (img_w, 60), (0,0,0), -1)
            cv.putText(display_img, f"PHASE {current_phase_idx+1}: {current_phase['name']}", (10, 25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(display_img, f"Count: {phase_progress}/{SAMPLES_PER_PHASE}", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if hand:
                hand_data = hand[0]
                
                # 1. Capture Raw 3D Coords (This is what we save)
                lmCoords = hand_data["lmCoords_3D"]
                
                # 2. Handle Bounding Box for Display
                # Since 'hand_data' is from the RAW image, we must flip the BBox coordinates
                # to draw them correctly on the FLIPPED 'display_img'
                raw_x, raw_y, w, h = hand_data.get("bbox", [0,0,0,0]) 
                disp_x = img_w - raw_x - w # Flip X-axis logic
                
                # --- Calculate Distance (Area) ---
                area = w * h
                dist_status = "Close" if area > CLOSE_THRESHOLD else "Far"
                
                # Check if distance matches target (Visual Feedback only)
                is_dist_correct = (dist_status == current_phase["target_dist"])
                status_color = (0, 255, 0) if is_dist_correct else (0, 255, 255)
                
                # Draw Box (Using flipped coordinates)
                cv.rectangle(display_img, (disp_x, raw_y), (disp_x+w, raw_y+h), status_color, 2)
                
                # Display Status
                status_text = f"Dist: {dist_status} ({area})"
                cv.putText(display_img, status_text, (disp_x, raw_y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Distance Warning
                if not is_dist_correct:
                    target = current_phase['target_dist']
                    msg = "MOVE CLOSER" if target == "Close" else "MOVE BACK"
                    cv.putText(display_img, f"{msg}", (10, 100), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # --- Recording Logic ---
                if is_recording:
                    gs_deque.append(lmCoords) # Saving the RAW, CORRECT coordinates
                    
                    # Visual: Red Recording Indicator
                    cv.circle(display_img, (img_w-50, 40), 20, (0, 0, 255), -1)
                    cv.putText(display_img, "REC", (img_w-65, 45), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                if is_recording:
                    cv.putText(display_img, "HAND LOST!", (10, 200), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv.imshow("Custom Dataset Recorder", display_img)
            key = cv.waitKey(1)

            # --- Controls ---
            if key == 27: # ESC
                break
            elif key == 32: # SPACE
                if not is_recording:
                    is_recording = True
                    gs_deque.clear()
                else:
                    is_recording = False
                    if len(gs_deque) > 15:
                        folder_name = f"{args.label}_{total_sample_count:03d}"
                        save_path = DATA_DIR / folder_name
                        save_path.mkdir(parents=True, exist_ok=True)
                        np.save(f"{save_path}/gs_sequence.npy", np.array(gs_deque))
                        
                        print(f">> Saved #{total_sample_count} ({current_phase['name']})")
                        total_sample_count += 1
                    else:
                        print(">> Too short! Discarded.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\n>> Closing Renderer and Exiting...")
        fusion_process.terminate()
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()