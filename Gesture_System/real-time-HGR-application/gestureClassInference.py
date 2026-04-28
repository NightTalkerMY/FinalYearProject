import os
import json
import pathlib
import cv2 as cv # Still needed for some internal processing usually, but not for imshow
import numpy as np
import requests 
from fastai.vision.all import *
from datetime import datetime, timedelta
from _helperFunctions import multiDetailsParser, hgrLogger

# [init.args.deets.defaults]
class Arguments:
    def __init__(self):
        self.mv_orientations = ["custom", "top-down", "front-away"]
        self.itr_scl_sizes = "null"
        cfg = json.load(open("./allConfigs.jsonc"))
        self.images_directory = cfg["images_directory"]
        self.hgr_archive = cfg["hgr_archive"]
        self.hgr_log = cfg["hgr_log"]
        self.debug_mode = cfg["debug_mode"]
        self.cpu_mode = cfg["cpu_mode"]
        self.idle = False
        self.data_files = []

args = Arguments()
deets = multiDetailsParser()
defaults.device = torch.device("cpu" if args.cpu_mode else "cuda:0")
Path(args.hgr_archive).mkdir(exist_ok=True)

# [init.learner.object]
from _functionsClasses import attachMetrics, e2eTunerLossWrapper

if os.name == 'nt': 
    pathlib.PosixPath = pathlib.WindowsPath
    pkl_file = "./.sources/[0c02]-6G-[cm_td_fa].pkl"
else: 
    pathlib.WindowsPath = pathlib.PosixPath
    pkl_file = "./.sources/[bf75]-7G-[cm_td_fa]-Linux.pkl"

attachMetrics(e2eTunerLossWrapper, args.mv_orientations, rename=True)
learn = load_learner(fname=pkl_file, cpu=args.cpu_mode)
print(f"INFO: Learner loaded. Vocabulary: {learn.dls.vocab}")

# [gesture.inference]
def _archive():
    for gs_tag in Path(args.images_directory).iterdir():
        gs_tag.replace(f"{args.hgr_archive}/{gs_tag.name}")

def main():
    data_files = L(dict.fromkeys([f.parent for f in get_image_files(args.images_directory)]))

    if data_files:
        print(f"INFO: Processing {len(data_files)} files...")
        args.idle = False

        data_dl = learn.dls.test_dl(data_files)
        preds, targs, decoded = learn.get_preds(dl=data_dl, with_decoded=True)
        decoded = np.array([i.numpy() for i in decoded]).tolist()
        aggregate = np.array([i.numpy() for i in preds]).sum(axis=0).argmax(axis=1).tolist()

        for idx, gs in enumerate(data_dl.items):
            # 1. Get Prediction String
            predicted_class = learn.dls.vocab[aggregate[idx]] 
            confidence = np.max(decoded[0]) 

            text = f">HGR: {gs.stem} -> [{predicted_class}]"
            print(f"👀 SAW: {predicted_class}") 

            # --- [BRIDGE TO ORCHESTRATOR] ---
            command = None
            if predicted_class == "swipeR": command = "swipe_right"
            elif predicted_class == "swipeL": command = "swipe_left"
            elif predicted_class == "Grab": command = "grab"
            elif predicted_class == "Expand": command = "expand"
            elif predicted_class == "SwipeU": command = "swipe_up"
            elif predicted_class == "SwipeD": command = "swipe_down"

            if command:
                print(f"🚀 MAPPED TO: {command} -> SENDING...")
                try:
                    requests.post(
                        "http://127.0.0.1:5000/gesture_command", 
                        json={"command": command, "confidence": float(confidence)},
                        timeout=0.5
                    )
                except Exception as e:
                    print(f"❌ Orchestrator unreachable: {e}")
            else:
                print(f"⚠️ IGNORED: {predicted_class} (No mapping)")
            
            hgrLogger(text, log=args.hgr_log)
        _archive()

    else:
        # if not (args.idle) and args.debug_mode:
        #     print("INFO: Waiting for data files...")
        # args.idle = True
        # time.sleep(0.5) 
        if not (args.idle) and args.debug_mode:
            print("INFO: Waiting for data files...")
        args.idle = True
        
        # --- NEW: KEEP WARM HEARTBEAT ---
        # Every 5 seconds, send a blank tensor through the model
        # This prevents Windows from paging RAM and the GPU from sleeping
        current_time = time.time()
        if not hasattr(args, 'last_warmup'): args.last_warmup = current_time
        
        if current_time - args.last_warmup > 5.0:
            try:
                with torch.no_grad(): # Don't track gradients
                    # Create a dummy image tensor (1 image, 3 channels, 224x224)
                    # Adjust the 224x224 to match your specific image size if different
                    dummy_input = torch.zeros(1, 3, 224, 224).to(defaults.device)
                    _ = learn.model(dummy_input)
                args.last_warmup = current_time
            except Exception:
                pass # Fail silently if the shape is slightly off, the attempt still wakes the GPU
                
        time.sleep(0.5)

if __name__ == "__main__":
    _archive()
    print("INFO: Initialized <gestureClassInference.py> (Backend Mode)...")
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print("INFO: Exiting...")