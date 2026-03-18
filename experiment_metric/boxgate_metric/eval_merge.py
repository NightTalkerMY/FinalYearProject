# eval_merge.py
# ------------------------------------------------------------
# OBJECTIVE 2 - GUIDED DATA COLLECTION
#
# Run only 4 times:
#
#   python eval_merge.py --scenario wrist_trigger
#   python eval_merge.py --scenario scale_trigger
#   python eval_merge.py --scenario normal
#   python eval_merge.py --scenario dim
#
# For each guided gesture:
#   - perform the gesture ONCE
#   - proposed sample is auto-saved by BoxGate
#   - baseline sample is saved from the SAME gesture instance by pressing SPACE
#
# Saves:
#   ./metric_tests/baseline_manual/<scenario>/<trial_id>/gs_sequence.npy
#   ./metric_tests/baseline_manual/<scenario>/<trial_id>/meta.json
#
#   ./metric_tests/proposed_auto/<scenario>/<trial_id>/gs_sequence.npy
#   ./metric_tests/proposed_auto/<scenario>/<trial_id>/meta.json
#
# Metadata includes:
#   - gesture_label
#   - scenario
#   - lighting
#   - trial_index
#   - round_index
#   - method
#
# Keyboard:
#   SPACE -> save baseline AFTER proposed has finished
#   c     -> clear/reset current gesture
#   n     -> skip current gesture
#   ESC   -> quit
# ------------------------------------------------------------

import re
import json
import time
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import starmap, repeat

import cv2 as cv
import numpy as np

from _mediapipePoseEstimation import HandDetector
from hgr_box_gate import BoxGate


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEFAULT_CFG = {
    "MAX_HISTORY": 150,
    "HISTORY_BUFFER": 20,
    "LOGGER_THRESHOLD": 0.2,
    "FRAME_SIZE": 640,
    "VERBOSE": False,
}

try:
    cfg = json.load(open("./allConfigs.jsonc"))
except Exception:
    print("WARNING: Config file not found, using defaults.")
    cfg = DEFAULT_CFG.copy()

BASE_SAVE_ROOT = Path("./metric_tests/baseline_manual")
PROP_SAVE_ROOT = Path("./metric_tests/proposed_auto")
BASE_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
PROP_SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# GESTURE SCHEDULES
# IMPORTANT: must match classifier vocab exactly
# ------------------------------------------------------------
WRIST_GESTURES = ["SwipeU", "SwipeD", "swipeL", "swipeR"]
SCALE_GESTURES = ["Grab", "Expand"]
ALL_GESTURES = WRIST_GESTURES + SCALE_GESTURES


def build_schedule(scenario: str, trials_per_gesture: int) -> list[dict]:
    """
    Returns grouped ordering:
      SwipeU x5, then SwipeD x5, then swipeL x5, ...
    """
    scenario = scenario.lower()

    if scenario == "wrist_trigger":
        gesture_set = WRIST_GESTURES
        lighting = "normal"
        group_type = "trigger"
    elif scenario == "scale_trigger":
        gesture_set = SCALE_GESTURES
        lighting = "normal"
        group_type = "trigger"
    elif scenario == "normal":
        gesture_set = ALL_GESTURES
        lighting = "normal"
        group_type = "lighting"
    elif scenario == "dim":
        gesture_set = ALL_GESTURES
        lighting = "dim"
        group_type = "lighting"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    schedule = []
    trial_counter = 1

    # grouped by gesture class
    for gesture in gesture_set:
        for round_idx in range(1, trials_per_gesture + 1):
            schedule.append({
                "trial_index": trial_counter,
                "round_index": round_idx,
                "gesture_label": gesture,
                "scenario": scenario,
                "lighting": lighting,
                "group_type": group_type,
            })
            trial_counter += 1

    return schedule


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def make_trial_id() -> str:
    return re.sub("[-:]", "", str(datetime.now())).replace(" ", ".")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_trial_dir(method_root: Path, scenario: str, trial_id: str) -> Path:
    save_path = method_root / scenario / trial_id
    ensure_dir(save_path)
    return save_path


def save_trial(
    method_root: Path,
    scenario: str,
    trial_id: str,
    gs_sequence: np.ndarray,
    meta: dict,
) -> Path:
    save_dir = build_trial_dir(method_root, scenario, trial_id)
    np.save(save_dir / "gs_sequence.npy", gs_sequence)
    save_json(save_dir / "meta.json", meta)
    return save_dir


def deque_to_array(dq: deque) -> np.ndarray:
    if len(dq) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.array(list(dq))


def current_timestamp_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clear_capture_state(base_deque: deque, prop_deque: deque, pre_buffer: deque, gate: BoxGate):
    base_deque.clear()
    prop_deque.clear()
    pre_buffer.clear()
    gate.reset()


def pretty_gesture_name(label: str) -> str:
    display_map = {
        "SwipeU": "Swipe Up",
        "SwipeD": "Swipe Down",
        "swipeL": "Swipe Left",
        "swipeR": "Swipe Right",
        "Grab": "Grab",
        "Expand": "Expand",
    }
    return display_map.get(label, label)


def infer_trigger_family(gesture_label: str) -> str:
    if gesture_label in WRIST_GESTURES:
        return "wrist_trigger"
    if gesture_label in SCALE_GESTURES:
        return "scale_trigger"
    return "unknown"


# ------------------------------------------------------------
# SAVE FUNCTIONS
# ------------------------------------------------------------
def save_baseline_trial(
    scenario: str,
    base_source,
    base_length: int,
    trial_plan: dict,
) -> Path | None:
    """
    base_source can be either:
    - deque: live rolling buffer
    - list: frozen snapshot buffer
    """
    if base_source is None or len(base_source) == 0:
        print("[BASELINE] No frames to save.")
        return None

    trial_id = make_trial_id()
    n_skeletons = min(len(base_source), base_length)
    gs = np.array(list(base_source)[-n_skeletons:])

    meta = {
        "trial_id": trial_id,
        "method": "baseline_manual",
        "scenario": trial_plan["scenario"],
        "lighting": trial_plan["lighting"],
        "group_type": trial_plan["group_type"],
        "gesture_label": trial_plan["gesture_label"],
        "trigger_family": infer_trigger_family(trial_plan["gesture_label"]),
        "trial_index": trial_plan["trial_index"],
        "round_index": trial_plan["round_index"],
        "saved_at": current_timestamp_str(),
        "n_frames": int(len(gs)),
        "valid_capture": True,
        "manual_stop": True,
        "auto_stop": False,
    }

    save_dir = save_trial(BASE_SAVE_ROOT, scenario, trial_id, gs, meta)
    print(f"[BASELINE] Saved {len(gs)} frames -> {save_dir}")
    return save_dir


def save_proposed_trial(
    scenario: str,
    prop_deque: deque,
    trial_plan: dict,
) -> Path | None:
    if len(prop_deque) == 0:
        print("[PROPOSED] No frames to save.")
        return None

    trial_id = make_trial_id()
    gs = deque_to_array(prop_deque)

    meta = {
        "trial_id": trial_id,
        "method": "proposed_auto",
        "scenario": trial_plan["scenario"],
        "lighting": trial_plan["lighting"],
        "group_type": trial_plan["group_type"],
        "gesture_label": trial_plan["gesture_label"],
        "trigger_family": infer_trigger_family(trial_plan["gesture_label"]),
        "trial_index": trial_plan["trial_index"],
        "round_index": trial_plan["round_index"],
        "saved_at": current_timestamp_str(),
        "n_frames": int(len(gs)),
        "valid_capture": True,
        "manual_stop": False,
        "auto_stop": True,
    }

    save_dir = save_trial(PROP_SAVE_ROOT, scenario, trial_id, gs, meta)
    print(f"[PROPOSED] Saved {len(gs)} frames -> {save_dir}")
    return save_dir


# ------------------------------------------------------------
# MAIN COLLECTION LOOP
# ------------------------------------------------------------
def live_stream_merged(nD: str, scenario: str, trials_per_gesture: int) -> None:
    time.sleep(1)
    nD = nD.upper()
    assert nD in {"2D", "3D"}, "nD must be '2d' or '3d'"

    schedule = build_schedule(scenario, trials_per_gesture)
    total_pairs = len(schedule)

    # ---- Baseline rolling buffer ----
    base_deque = deque(maxlen=cfg["MAX_HISTORY"])
    base_length = cfg["MAX_HISTORY"] - cfg["HISTORY_BUFFER"]
    base_minimum = int(cfg["LOGGER_THRESHOLD"] * cfg["MAX_HISTORY"])

    # ---- Proposed buffers ----
    prop_deque = deque(maxlen=cfg["MAX_HISTORY"])
    pre_buffer = deque(maxlen=6)

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["FRAME_SIZE"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(cfg["FRAME_SIZE"] * 9 / 16))

    window_name = "OBJECTIVE 2 COLLECTION: BASELINE VS PROPOSED"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, 50, 15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()

    schedule_idx = 0

    # Shared-capture state
    proposed_saved = False
    baseline_ready_to_save = False
    last_finished_gesture = None
    frozen_baseline_buffer = None

    print(f"INFO: Scenario = '{scenario}'")
    print(f"INFO: Trials per gesture = {trials_per_gesture}")
    print(f"INFO: Total gesture pairs = {total_pairs}")
    print("INFO: Gesture order is grouped by class, e.g. Swipe Up x5, then Swipe Down x5.")
    print("INFO: For each guided gesture, perform it ONCE only.")
    print("INFO: Proposed will auto-save first, then press SPACE to save baseline from the SAME gesture instance.")
    print("Keyboard: SPACE=save baseline after proposed finish | c=reset | n=skip | ESC=quit")

    while True:
        if schedule_idx >= total_pairs:
            print("\n>>> ALL COLLECTIONS COMPLETE <<<")
            break

        trial_plan = schedule[schedule_idx]
        gesture_label = trial_plan["gesture_label"]

        success, img = cap.read()
        if not success:
            continue

        img = cv.flip(img, 1)
        hand_data, img = detector.findHands(img, draw=True)

        prop_state = "NO_HAND"

        # --------------------------------------------------------
        # SHARED CAPTURE
        # --------------------------------------------------------
        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]

            # Keep rolling baseline until proposed finishes and snapshot is frozen
            if not baseline_ready_to_save:
                base_deque.append(lmCoords)

            # Proposed gate active until saved
            if not proposed_saved:
                prop_state, _ = gate.process(lmCoords)
                pre_buffer.append(lmCoords)

                if prop_state == "RECORDING":
                    if gate.frame_count == 1:
                        prop_deque.clear()
                        prop_deque.extend(pre_buffer)
                        print(f">>> PROPOSED STARTED for {gesture_label}")

                    prop_deque.append(lmCoords)

                elif prop_state == "FINISHED":
                    save_proposed_trial(scenario, prop_deque, trial_plan)

                    proposed_saved = True
                    baseline_ready_to_save = True
                    last_finished_gesture = gesture_label

                    # Freeze baseline immediately so user reaction time does not change baseline clip
                    frozen_baseline_buffer = list(base_deque)

                    prop_deque.clear()
                    pre_buffer.clear()
                    gate.reset()

                    # blue flash = proposed saved
                    flash = img.copy()
                    flash[:] = (255, 0, 0)
                    cv.imshow(window_name, flash)
                    cv.waitKey(100)

                    print(f"INFO: Proposed saved for {gesture_label}. Baseline snapshot frozen. Press SPACE now.")

        else:
            if not proposed_saved:
                gate.reset()
                pre_buffer.clear()

        # --------------------------------------------------------
        # UI
        # --------------------------------------------------------
        pair_num = trial_plan["trial_index"]
        round_num = trial_plan["round_index"]
        trigger_family = infer_trigger_family(gesture_label)

        cv.putText(
            img,
            f"Scenario: {scenario} | Lighting: {trial_plan['lighting']}",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv.putText(
            img,
            f"Gesture Pair {pair_num}/{total_pairs} | Repeat {round_num}/{trials_per_gesture}",
            (10, 55),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv.putText(
            img,
            f"Target Gesture: {pretty_gesture_name(gesture_label)}",
            (10, 85),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv.putText(
            img,
            f"Trigger Family: {trigger_family}",
            (10, 115),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 255, 200),
            2,
        )

        if not proposed_saved:
            cv.putText(
                img,
                "NOW: Perform the gesture once. Proposed is waiting to auto-capture.",
                (10, 155),
                cv.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 165, 255),
                2,
            )
            cv.putText(
                img,
                f"Proposed state: {prop_state}",
                (10, 185),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255) if prop_state == "RECORDING" else (0, 255, 0),
                2,
            )
        else:
            cv.putText(
                img,
                "NOW: Proposed saved. Press SPACE to save baseline from the SAME gesture.",
                (10, 155),
                cv.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 0, 255),
                2,
            )
            cv.putText(
                img,
                f"Frozen baseline ready for: {pretty_gesture_name(last_finished_gesture)}",
                (10, 185),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv.putText(
            img,
            f"Live baseline buffer: {len(base_deque)}",
            (10, 215),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv.putText(
            img,
            f"Proposed buffer: {len(prop_deque)}",
            (10, 245),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

        cv.imshow(window_name, img)
        key = cv.waitKey(1)

        # --------------------------------------------------------
        # Keyboard controls
        # --------------------------------------------------------
        if key == 27:  # ESC
            break

        elif key == ord("c"):
            clear_capture_state(base_deque, prop_deque, pre_buffer, gate)
            proposed_saved = False
            baseline_ready_to_save = False
            last_finished_gesture = None
            frozen_baseline_buffer = None
            print("INFO: Cleared buffers and reset current guided trial.")

        elif key == ord("n"):
            clear_capture_state(base_deque, prop_deque, pre_buffer, gate)
            proposed_saved = False
            baseline_ready_to_save = False
            last_finished_gesture = None
            frozen_baseline_buffer = None
            schedule_idx += 1
            print("INFO: Skipped current gesture pair -> moving to next.")

        elif key == 32:  # SPACE = save baseline after proposed finish
            if baseline_ready_to_save:
                if frozen_baseline_buffer is not None and len(frozen_baseline_buffer) > base_minimum:
                    save_baseline_trial(scenario, frozen_baseline_buffer, base_length, trial_plan)

                    clear_capture_state(base_deque, prop_deque, pre_buffer, gate)
                    proposed_saved = False
                    baseline_ready_to_save = False
                    last_finished_gesture = None
                    frozen_baseline_buffer = None
                    schedule_idx += 1

                    # red flash = baseline saved and trial complete
                    flash = img.copy()
                    flash[:] = (0, 0, 255)
                    cv.imshow(window_name, flash)
                    cv.waitKey(100)

                    print(f"INFO: Completed shared capture for {gesture_label}. Moving to next gesture.")
                else:
                    print("WARNING: Frozen baseline buffer too short. Retry this gesture.")
            else:
                print("INFO: SPACE ignored. Wait until proposed auto-save finishes first.")

    cap.release()
    cv.destroyAllWindows()


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Objective 2 guided metric collection")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["wrist_trigger", "scale_trigger", "normal", "dim"],
        help="Collection scenario"
    )
    parser.add_argument(
        "--trials_per_gesture",
        type=int,
        default=5,
        help="Number of trials per gesture class"
    )
    parser.add_argument(
        "--nd",
        type=str,
        default="3d",
        choices=["2d", "3d", "2D", "3D"],
        help="Use 2D or 3D landmarks"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    live_stream_merged(
        nD=args.nd,
        scenario=args.scenario,
        trials_per_gesture=args.trials_per_gesture,
    )