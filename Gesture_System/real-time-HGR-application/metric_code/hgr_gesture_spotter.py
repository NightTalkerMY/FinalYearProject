# hgr_gesture_spotter.py
# ------------------------------------------------------------
# REFERENCE IMPLEMENTATION: Gesture Spotter Single-Time Activation
#
# Based on: Shen et al., "Gesture Spotter: A Rapid Prototyping Tool
# for Key Gesture Spotting in Virtual and Augmented Reality
# Applications", IEEE TVCG 2022.
#
# Implements the sliding window + single-time activation approach
# from Section 3.2 of the paper. Uses motion-based scoring as a
# lightweight proxy for the A-ResLSTM classifier since the goal
# is to compare the SEGMENTATION approach, not the classifier.
#
# The key idea:
#   1. Maintain a sliding window of length `l` with step `s`
#   2. Score each window (proxy for classifier confidence)
#   3. If score > detection_threshold for `recurrence_threshold`
#      consecutive windows -> fire a single-time activation
#   4. The captured segment is the fixed-length window
#
# Interface mirrors BoxGate: process(lmCoords) -> (state, score)
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
from collections import deque


class GestureSpotterGate:
    """
    Sliding window + single-time activation gate.

    Processes skeleton frames one at a time and fires an activation
    when a gesture is detected via the two-threshold mechanism
    described in the Gesture Spotter paper (Section 3.2):

        - detection_threshold: minimum score to count as a detection
        - recurrence_threshold: consecutive detections needed to fire

    States returned by process():
        FILLING   - sliding window buffer not yet full
        IDLE      - between slide steps (no scoring this frame)
        DETECTING - scored but below activation criteria
        ACTIVATED - gesture detected, single-time activation fired
        COOLDOWN  - post-activation suppression period
    """

    def __init__(
        self,
        window_length: int = 30,
        slide_step: int = 1,
        detection_threshold: float = 0.8,
        recurrence_threshold: int = 3,
        cooldown_frames: int = 20,
        motion_floor: float = 0.005,
    ):
        # --- Paper parameters (Section 3.2) ---
        self.window_length = window_length          # l
        self.slide_step = slide_step                # s
        self.detection_threshold = detection_threshold
        self.recurrence_threshold = recurrence_threshold
        self.cooldown_frames = cooldown_frames

        # --- Scoring parameters ---
        self.motion_floor = motion_floor  # absolute minimum wrist motion
        self.scale_floor = motion_floor * 0.2  # scale changes are ~5x smaller
        self.min_gesture_frames = 8  # matches BoxGate MIN_FRAMES

        # --- Internal state ---
        self.buffer: deque = deque(maxlen=window_length)
        self.frame_idx: int = 0
        self.consecutive_count: int = 0
        self.last_detected_class: str | None = None  # tracks dominant channel
        self.cooldown_counter: int = 0
        self.activated_window: np.ndarray | None = None

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------
    def process(self, lmCoords: np.ndarray) -> tuple[str, float]:
        """
        Feed one skeleton frame.  Returns (state, score).
        Same call convention as BoxGate.process().
        """
        self.buffer.append(lmCoords.copy())
        self.frame_idx += 1

        # Need a full window before scoring
        if len(self.buffer) < self.window_length:
            return "FILLING", 0.0

        # Post-activation suppression (prevent double-fire)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "COOLDOWN", 0.0

        # Only score every `slide_step` frames
        if self.frame_idx % self.slide_step != 0:
            return "IDLE", 0.0

        # --- Score the current sliding window ---
        window = np.array(list(self.buffer))
        score, detected_class = self._score_window(window)

        # --- Single-time activation algorithm (Section 3.2) ---
        # Paper: "the same gesture detected in a row" must exceed
        # recurrence threshold. We reset if dominant channel changes.
        if score >= self.detection_threshold:
            if detected_class == self.last_detected_class:
                self.consecutive_count += 1
            else:
                self.consecutive_count = 1
                self.last_detected_class = detected_class
        else:
            self.consecutive_count = 0
            self.last_detected_class = None

        if self.consecutive_count >= self.recurrence_threshold:
            # ACTIVATION - save the window and enter cooldown
            self.activated_window = window.copy()
            self.consecutive_count = 0
            self.cooldown_counter = self.cooldown_frames
            return "ACTIVATED", score

        return "DETECTING", score

    def get_activated_window(self) -> np.ndarray | None:
        """Return the window that triggered the last activation."""
        return self.activated_window

    def reset(self):
        """Reset all state (call when hand is lost)."""
        self.buffer.clear()
        self.frame_idx = 0
        self.consecutive_count = 0
        self.last_detected_class = None
        self.cooldown_counter = 0
        self.activated_window = None

    # ----------------------------------------------------------
    # SCORING
    # ----------------------------------------------------------
    def _score_window(self, window: np.ndarray) -> tuple[float, str | None]:
        """
        Compute a gesture-presence score for the sliding window.

        This is a motion-energy proxy for the A-ResLSTM classifier
        output.  Wrist velocity and fingertip scale change are scored
        as **independent channels** (mirroring BoxGate's separate
        swipe/grab triggers), then the max is taken so that either
        motion type can drive detection.

        Returns (score, detected_class):
            score: float in [0, 1]
            detected_class: "wrist" | "scale" | None (dominant channel)
        """
        # Wrist trajectory (joint 0)
        wrist = window[:, 0, :]                                     # (l, 3)
        wrist_vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)  # (l-1,)

        # Fingertip scale change (joints 4,8,12,16,20)
        tips = window[:, [4, 8, 12, 16, 20], :]                    # (l, 5, 3)
        scales = np.mean(
            np.linalg.norm(tips - wrist[:, np.newaxis, :], axis=2),
            axis=1,
        )                                                           # (l,)
        scale_vel = np.abs(np.diff(scales))                         # (l-1,)

        # Score each channel independently so either can drive detection
        wrist_score = self._channel_score(wrist_vel, self.motion_floor)
        scale_score = self._channel_score(scale_vel, self.scale_floor)

        if wrist_score >= scale_score:
            return wrist_score, ("wrist" if wrist_score > 0 else None)
        else:
            return scale_score, ("scale" if scale_score > 0 else None)

    def _channel_score(self, signal: np.ndarray, floor: float) -> float:
        """
        Score a single motion channel.

        Instead of dividing active frames by total window length
        (which punishes short gestures like grab/expand), the score
        saturates once `min_gesture_frames` active frames are present.

        This approximates how the A-ResLSTM + attention mechanism
        would confidently detect a gesture even if it occupies only
        a portion of the sliding window.

            dynamic_thresh = max(peak * 0.15, floor)
            active = count(signal > dynamic_thresh)
            score  = min(1.0, active / min_gesture_frames)
        """
        if len(signal) == 0:
            return 0.0

        peak = float(np.max(signal))
        if peak < floor:
            return 0.0

        dynamic_thresh = max(peak * 0.15, floor)
        active_count = int(np.sum(signal > dynamic_thresh))

        # Saturate once enough active frames confirm a real gesture
        return min(1.0, active_count / self.min_gesture_frames)
