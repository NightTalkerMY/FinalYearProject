# # e2eET Skeleton Based HGR Using Data-Level Fusion
# # Dynamic Hand Gestures Classification: Live Stream Demo
# # ---------------------------------------------------------

# import os
# import json
# import pathlib
# import cv2 as cv
# import numpy as np
# import requests # <--- IMPORTANT: Needed to talk to Orchestrator
# from fastai.vision.all import *
# from datetime import datetime, timedelta

# from _helperFunctions import multiDetailsParser, hgrLogger


# # [init.args.deets.defaults]
# class Arguments:
#     def __init__(self):
#         self.mv_orientations = ["custom", "top-down", "front-away"]
#         self.itr_scl_sizes = "null"

#         cfg = json.load(open("./allConfigs.jsonc"))
#         self.images_directory = cfg["images_directory"]
#         self.hgr_archive = cfg["hgr_archive"]
#         self.hgr_log = cfg["hgr_log"]
#         self.debug_mode = cfg["debug_mode"]
#         self.cpu_mode = cfg["cpu_mode"]

#         self.idle = False
#         self.data_files = []


# args = Arguments()
# deets = multiDetailsParser()
# defaults.device = torch.device("cpu" if args.cpu_mode else "cuda:0")
# Path(args.hgr_archive).mkdir(exist_ok=True)


# # [init.learner.object]
# from _functionsClasses import attachMetrics, e2eTunerLossWrapper

# if os.name == 'nt': 
#     pathlib.PosixPath = pathlib.WindowsPath
#     pkl_file = "./.sources/[0c02]-6G-[cm_td_fa].pkl"
# else: 
#     pathlib.WindowsPath = pathlib.PosixPath
#     pkl_file = "./.sources/[bf75]-7G-[cm_td_fa]-Linux.pkl"

# attachMetrics(e2eTunerLossWrapper, args.mv_orientations, rename=True)
# learn = load_learner(fname=pkl_file, cpu=args.cpu_mode)
# str_dls_vocab = " ".join([f"{i}.{v}" for i, v in enumerate(learn.dls.vocab)])
# print(f"INFO: dls.vocab=[{str_dls_vocab}]")


# # [display.inferences]
# window_name = "gestureClassInference.py"

# def _display(text):
#     _display = np.zeros((240, 720, 3))
#     o_x, o_y = 25, 50
#     for line in text.splitlines():
#         x = o_x
#         if "\t" in line:
#             x += 50
#             line = line.replace("\t", "")
#         cv.putText(_display, line, (x, o_y), cv.FONT_HERSHEY_SIMPLEX, 0.625, (0, 255, 0), 1)
#         o_y += 35
#     cv.imshow(window_name, _display)
#     cv.waitKey(100)


# # [gesture.inference]
# def _archive():
#     for gs_tag in Path(args.images_directory).iterdir():
#         gs_tag.replace(f"{args.hgr_archive}/{gs_tag.name}")


# def _inference_time(gs_tag):
#     gs_time = [int(gs_tag[9:15][i : i + 2]) for i in (0, 2, 4)]
#     gs_time = timedelta(hours=gs_time[0], minutes=gs_time[1], seconds=gs_time[2])
#     return str(datetime.now() - gs_time)[11:19]


# def main():
#     data_files = L(dict.fromkeys([f.parent for f in get_image_files(args.images_directory)]))

#     if data_files:
#         print(f"INFO: Processing {len(data_files)} files...")
#         args.idle = False

#         data_dl = learn.dls.test_dl(data_files)
#         preds, targs, decoded = learn.get_preds(dl=data_dl, with_decoded=True)
#         decoded = np.array([i.numpy() for i in decoded]).tolist()
#         aggregate = np.array([i.numpy() for i in preds]).sum(axis=0).argmax(axis=1).tolist()

#         for idx, gs in enumerate(data_dl.items):
#             # 1. Get Prediction String
#             predicted_class = learn.dls.vocab[aggregate[idx]] 
#             confidence = np.max(decoded[0]) 

#             text = f">HGR: {gs.stem} -> [{predicted_class}]"
            
#             # --- [BRIDGE TO ORCHESTRATOR] ---
#             command = None
            
#             # IMPORTANT: Adjust these strings to match YOUR model's exact labels
#             if "Swipe Right" in predicted_class or "SwipeRight" in predicted_class: 
#                 command = "NEXT"
#             elif "Swipe Left" in predicted_class or "SwipeLeft" in predicted_class:
#                 command = "PREV"
#             elif "Grab" in predicted_class or "Fist" in predicted_class:
#                 command = "SELECT"
#             elif "Palm" in predicted_class or "Open" in predicted_class or "Expand" in predicted_class:
#                 command = "EXIT"

#             if command:
#                 print(f"🚀 SENDING GESTURE COMMAND: {command}")
#                 try:
#                     requests.post(
#                         "http://127.0.0.1:5000/gesture_command", 
#                         json={"command": command, "confidence": float(confidence)},
#                         timeout=0.5
#                     )
#                 except Exception as e:
#                     print(f"❌ Orchestrator unreachable: {e}")
#             # ---------------------------------

#             hgrLogger(text, log=args.hgr_log)

#         _archive()

#     else:
#         text = "INFO: Waiting for data files..."
#         if not (args.idle) and args.debug_mode:
#             print(text)
#         args.idle = True


# if __name__ == "__main__":
#     _archive()
#     print("INFO: Initialized <gestureClassInference.py> ...")

#     try:
#         while True:
#             main()
#     except KeyboardInterrupt:
#         cv.destroyAllWindows()
#         print("INFO: KeyboardInterrupt received. Exiting...")


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
        if not (args.idle) and args.debug_mode:
            print("INFO: Waiting for data files...")
        args.idle = True
        time.sleep(0.5) 

if __name__ == "__main__":
    _archive()
    print("INFO: Initialized <gestureClassInference.py> (Backend Mode)...")
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print("INFO: Exiting...")