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
from hgr_gesture_spotter import GestureSpotterGate


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
REF_SAVE_ROOT  = Path("./metric_tests/reference_spotter")
BASE_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
PROP_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
REF_SAVE_ROOT.mkdir(parents=True, exist_ok=True)


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


def save_reference_trial(
    scenario: str,
    ref_window: np.ndarray,
    trial_plan: dict,
    spotter: GestureSpotterGate,
) -> Path | None:
    if ref_window is None or len(ref_window) == 0:
        print("[REFERENCE] No frames to save.")
        return None

    trial_id = make_trial_id()
    gs = ref_window

    meta = {
        "trial_id": trial_id,
        "method": "reference_spotter",
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
        "detection_method": "sliding_window_activation",
        "window_length": spotter.window_length,
        "slide_step": spotter.slide_step,
        "detection_threshold": spotter.detection_threshold,
        "recurrence_threshold": spotter.recurrence_threshold,
    }

    save_dir = save_trial(REF_SAVE_ROOT, scenario, trial_id, gs, meta)
    print(f"[REFERENCE] Saved {len(gs)} frames -> {save_dir}")
    return save_dir


def save_missed_trial(method: str, scenario: str, trial_plan: dict) -> None:
    """Save a meta-only marker for a method that failed to trigger (no gs_sequence)."""
    root_map = {
        "proposed_auto": PROP_SAVE_ROOT,
        "reference_spotter": REF_SAVE_ROOT,
    }
    root = root_map.get(method)
    if root is None:
        return

    trial_id = make_trial_id()
    save_dir = build_trial_dir(root, scenario, trial_id)

    meta = {
        "trial_id": trial_id,
        "method": method,
        "scenario": trial_plan["scenario"],
        "lighting": trial_plan["lighting"],
        "group_type": trial_plan["group_type"],
        "gesture_label": trial_plan["gesture_label"],
        "trigger_family": infer_trigger_family(trial_plan["gesture_label"]),
        "trial_index": trial_plan["trial_index"],
        "round_index": trial_plan["round_index"],
        "saved_at": current_timestamp_str(),
        "n_frames": 0,
        "valid_capture": False,
        "missed": True,
    }

    save_json(save_dir / "meta.json", meta)
    print(f"[{method.upper()}] Missed trial saved -> {save_dir}")


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

    window_name = "OBJECTIVE 2 COLLECTION: BASELINE VS REFERENCE VS PROPOSED"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, 50, 15)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    gate = BoxGate()
    spotter = GestureSpotterGate()

    schedule_idx = 0

    # Shared-capture state (three methods, fully independent)
    proposed_saved = False
    reference_saved = False
    baseline_saved = False

    print(f"INFO: Scenario = '{scenario}'")
    print(f"INFO: Trials per gesture = {trials_per_gesture}")
    print(f"INFO: Total gesture pairs = {total_pairs}")
    print("INFO: Gesture order is grouped by class, e.g. Swipe Up x5, then Swipe Down x5.")
    print("INFO: For each guided gesture, perform it ONCE only.")
    print("INFO: All three methods run independently and in parallel.")
    print("INFO: Press SPACE whenever YOU think the gesture is done (baseline).")
    print("INFO: Auto-methods will trigger on their own.")
    print("Keyboard: SPACE=save baseline (2nd SPACE=force-advance) | c=reset | n=skip | ESC=quit")

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
        ref_state = "NO_HAND"
        ref_score = 0.0

        # --------------------------------------------------------
        # SHARED CAPTURE (three methods)
        # --------------------------------------------------------
        if hand_data:
            hand = hand_data[0]
            lmCoords = hand[f"lmCoords_{nD}"]

            # Baseline always collects until user presses SPACE
            if not baseline_saved:
                base_deque.append(lmCoords)

            # ---- Proposed gate (BoxGate) ----
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

                    prop_deque.clear()
                    pre_buffer.clear()
                    gate.reset()

                    # blue flash = proposed saved
                    flash = img.copy()
                    flash[:] = (255, 0, 0)
                    cv.imshow(window_name, flash)
                    cv.waitKey(100)

                    print(f"INFO: Proposed saved for {gesture_label}.")

            # ---- Reference gate (Gesture Spotter) ----
            if not reference_saved:
                ref_state, ref_score = spotter.process(lmCoords)

                if ref_state == "ACTIVATED":
                    save_reference_trial(
                        scenario, spotter.get_activated_window(),
                        trial_plan, spotter,
                    )
                    reference_saved = True

                    # green flash = reference saved
                    flash = img.copy()
                    flash[:] = (0, 255, 0)
                    cv.imshow(window_name, flash)
                    cv.waitKey(100)

                    print(f"INFO: Reference saved for {gesture_label}.")

        else:
            if not proposed_saved:
                gate.reset()
                pre_buffer.clear()
            if not reference_saved:
                spotter.reset()

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
            0.55,
            (255, 255, 255),
            2,
        )

        cv.putText(
            img,
            f"Pair {pair_num}/{total_pairs} | Repeat {round_num}/{trials_per_gesture} | {trigger_family}",
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        cv.putText(
            img,
            f"Target: {pretty_gesture_name(gesture_label)}",
            (10, 75),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Dynamic instruction based on capture state
        all_done = proposed_saved and reference_saved and baseline_saved
        if all_done:
            instruction = "ALL 3 SAVED. Moving to next..."
            inst_color = (0, 255, 0)
        elif not baseline_saved:
            instruction = "Perform gesture. Press SPACE when YOU think it's done."
            inst_color = (0, 165, 255)
        else:
            instruction = "Baseline saved. SPACE again to force-advance (skip stuck methods)."
            inst_color = (0, 200, 200)

        cv.putText(
            img, instruction, (10, 105),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, inst_color, 2,
        )

        # Proposed status
        prop_display = "SAVED" if proposed_saved else prop_state
        prop_color = (0, 200, 0) if proposed_saved else (
            (0, 165, 255) if prop_state == "RECORDING" else (180, 180, 180)
        )
        cv.putText(
            img,
            f"Proposed: {prop_display} | buf={len(prop_deque)}",
            (10, 135),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, prop_color, 2,
        )

        # Reference status
        ref_display = "SAVED" if reference_saved else ref_state
        ref_color = (0, 200, 0) if reference_saved else (
            (0, 255, 255) if ref_state == "ACTIVATED" else (180, 180, 180)
        )
        cv.putText(
            img,
            f"Reference: {ref_display} | score={ref_score:.2f}",
            (10, 160),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, ref_color, 2,
        )

        # Baseline status
        base_display = "SAVED" if baseline_saved else f"{len(base_deque)} frames (rolling)"
        base_color = (0, 200, 0) if baseline_saved else (100, 100, 255)
        cv.putText(
            img,
            f"Baseline: {base_display}",
            (10, 185),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, base_color, 2,
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
            spotter.reset()
            proposed_saved = False
            reference_saved = False
            baseline_saved = False
            print("INFO: Cleared buffers and reset current guided trial.")

        elif key == ord("n"):
            clear_capture_state(base_deque, prop_deque, pre_buffer, gate)
            spotter.reset()
            proposed_saved = False
            reference_saved = False
            baseline_saved = False
            schedule_idx += 1
            print("INFO: Skipped current gesture pair -> moving to next.")

        elif key == 32:  # SPACE
            if not baseline_saved:
                # First SPACE: save baseline
                if len(base_deque) > base_minimum:
                    save_baseline_trial(scenario, list(base_deque), base_length, trial_plan)
                    baseline_saved = True

                    # red flash = baseline saved
                    flash = img.copy()
                    flash[:] = (0, 0, 255)
                    cv.imshow(window_name, flash)
                    cv.waitKey(100)

                    print(f"INFO: Baseline saved for {gesture_label}.")
                else:
                    print(f"WARNING: Baseline too short ({len(base_deque)} frames). Keep going.")
            else:
                # Second SPACE: force-advance (auto-methods missed this gesture)
                if not proposed_saved:
                    save_missed_trial("proposed_auto", scenario, trial_plan)
                if not reference_saved:
                    save_missed_trial("reference_spotter", scenario, trial_plan)
                # Force all done so auto-advance fires
                proposed_saved = True
                reference_saved = True
                print("INFO: Force-advancing to next trial.")

        # --------------------------------------------------------
        # Auto-advance when all three methods have saved
        # --------------------------------------------------------
        if proposed_saved and reference_saved and baseline_saved:
            clear_capture_state(base_deque, prop_deque, pre_buffer, gate)
            spotter.reset()
            proposed_saved = False
            reference_saved = False
            baseline_saved = False
            schedule_idx += 1
            print(f"INFO: All 3 methods saved for {gesture_label}. Auto-advancing.")

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