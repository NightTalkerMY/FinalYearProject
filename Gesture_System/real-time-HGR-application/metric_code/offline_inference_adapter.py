# offline_inference_adapter.py
# ------------------------------------------------------------
# OFFLINE ADAPTER FOR OBJECTIVE 2 INFERENCE
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import builtins
import json
import shutil
import tempfile
import pathlib
from pathlib import Path

# ============================================================
# THE MUZZLE: Prevent _functionsClasses.py from silently exiting
# ============================================================
def _prevent_exit(*args, **kwargs):
    print(f"\n[TRAP] _functionsClasses.py tried to exit with {args}. Ignored so evaluation can continue!")

sys.exit = _prevent_exit
builtins.exit = _prevent_exit
builtins.quit = _prevent_exit
# ============================================================

import numpy as np
from scipy import ndimage
import matplotlib.colors as mcolors

import vispy.io
from vispy import scene

from fastai.vision.all import load_learner, defaults, torch, L

# ---------------------------
# CONFIG
# ---------------------------
CFG = json.load(open("./allConfigs.jsonc"))

CPU_MODE = CFG["cpu_mode"]
defaults.device = torch.device("cpu" if CPU_MODE else "cuda:0")

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath
    PKL_FILE = "./[0c02]-6G-[cm_td_fa].pkl"
else:
    pathlib.WindowsPath = pathlib.PosixPath
    PKL_FILE = "./.sources/[bf75]-7G-[cm_td_fa]-Linux.pkl"

VALID_LABELS = {"swipeR", "swipeL", "SwipeU", "SwipeD", "Grab", "Expand"}
MV_ORIENTATIONS = ["custom", "top-down", "front-away"]

# ---------------------------
# PATCH __main__ EXPECTATIONS
# ---------------------------
class _CallableDummy:
    def __call__(self, *args, **kwargs):
        return _CallableDummy()
    def __getattr__(self, item):
        if "seed" in item.lower():
            return 42
        return _CallableDummy()
    def __int__(self):
        return 42
    def __index__(self):
        return 42
    def __bool__(self):
        return False  # If it checks 'if deets.flag:', safely evaluate to False
    def __str__(self):
        return "offline_dummy"

class _OfflineArgs:
    def __init__(self):
        self.mv_orientations = MV_ORIENTATIONS
        self.itr_scl_sizes = "null"
        self.cpu_mode = CPU_MODE
        self.debug_mode = False
        self.images_directory = CFG.get("images_directory", "./images")
        self.hgr_archive = CFG.get("hgr_archive", "./archive")
        self.hgr_log = CFG.get("hgr_log", False)
        self.idle = False
        self.data_files = []
        self.seed = 42

    def __getattr__(self, item):
        if "seed" in item.lower():
            return 42
        return _CallableDummy()

class _OfflineDeets:
    def __init__(self):
        self.e_tag = "offline_eval"
        self.e_secret = "offline_secret"
        self.seed = 42
        self.e_repr_seed = 42
        self.cpu_mode = CPU_MODE

    def __getattr__(self, item):
        if "seed" in item.lower():
            return 42
        return _CallableDummy()

main_mod = sys.modules.get("__main__")
if main_mod is not None:
    main_mod.args = _OfflineArgs()
    main_mod.deets = _OfflineDeets()

# ---------------------------
# LOAD LEARNER ONCE
# ---------------------------
from _functionsClasses import attachMetrics, e2eTunerLossWrapper

attachMetrics(e2eTunerLossWrapper, MV_ORIENTATIONS, rename=True)
LEARN = load_learner(fname=PKL_FILE, cpu=CPU_MODE)

VOCAB = list(LEARN.dls.vocab)
VOCAB_SET = set(VOCAB)

if not VALID_LABELS.issubset(VOCAB_SET):
    print("[WARN] Learner vocab differs from expected labels.")
    print("[WARN] Learner vocab:", VOCAB)

# ---------------------------
# DATA-LEVEL FUSION RENDERER
# ---------------------------
class OfflineFusionRenderer:
    def __init__(self):
        self.dhg1428_mode = CFG["dhg1428_mode"]
        prefix = "dhg" if self.dhg1428_mode else "mp"

        self.connection_map = CFG[f"{prefix}_connection_map"]
        self.finger_tips = CFG[f"{prefix}_finger_tips"]
        self.fingers_colors = dict(CFG[f"{prefix}_fingers_colors"])

        self.vo_temporal_gradations = CFG["add_vo_temporal_gradations"]
        self.vo_skeletons = CFG["add_vo_skeletons"]
        self.temporal_trails = CFG["temporal_trails"]

        self.w_visuals = CFG.setdefault("w_visuals", 3.5)
        self.sz_canvas = CFG.setdefault("sz_canvas", 960)

        self._build_canvas()
        self._build_visuals()
        self._make_colored_fingers()

    def _build_canvas(self):
        self.canvas = scene.SceneCanvas(
            keys=None,
            show=False,
            bgcolor="black",
            size=(self.sz_canvas, self.sz_canvas),
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 0.0
        self.view.camera.roll = 0.0
        self.view.camera.distance = 0.0
        
        self.canvas.show(visible=True)
        self.canvas.show(visible=False)
        self.canvas.app.process_events()

    def _build_visuals(self):
        self.skeleton = scene.visuals.Line()
        self.view.add(self.skeleton)

        self.skeleton_nodes = scene.visuals.Markers()
        self.view.add(self.skeleton_nodes)

        self.tips_visuals = {}
        if self.temporal_trails == "lines":
            for tip in self.finger_tips:
                self.tips_visuals[tip] = scene.visuals.Line()
                self.view.add(self.tips_visuals[tip])
        else:
            for tip in self.finger_tips:
                self.tips_visuals[tip] = scene.visuals.Markers()
                self.view.add(self.tips_visuals[tip])

    def _set_view(self, v_orientation: str, cam_scaling):
        self.view.camera.scale_factor = cam_scaling
        self.view.camera.center = [self.sz_canvas / 2] * 3

        if v_orientation == "top-down":
            self.view.camera.elevation, self.view.camera.azimuth = 0.0, 0.0
        elif v_orientation == "front-to":
            self.view.camera.elevation, self.view.camera.azimuth = 90.0, 180.0
        elif v_orientation == "front-away":
            self.view.camera.elevation, self.view.camera.azimuth = -90.0, 0.0
        elif v_orientation == "side-right":
            self.view.camera.elevation, self.view.camera.azimuth = 0.0, -90.0
        elif v_orientation == "side-left":
            self.view.camera.elevation, self.view.camera.azimuth = 0.0, 90.0
        elif v_orientation == "custom":
            self.view.camera.scale_factor = 0.85
            self.view.camera.elevation, self.view.camera.azimuth = 30.0, -132.5

    def _get_tip_colormap(self, tip, n_points: int):
        tipcolor = self.fingers_colors[tip]
        tipcolor = mcolors.to_rgb(mcolors.CSS4_COLORS[tipcolor])

        colormap = np.ones((n_points, 4))
        colormap[:, 0:3] = tipcolor
        colormap[:, 3] = (
            np.linspace(0, 1, n_points)
            if self.vo_temporal_gradations
            else np.linspace(1, 1, n_points)
        )
        return colormap

    def _make_colored_fingers(self):
        n = len(self.connection_map) * 2
        self.colors = np.ones((n, 4), dtype=np.float32)

        right_cm = [item[1] for item in self.connection_map]
        offset = 0

        for node in range(len(self.fingers_colors)):
            nodecolor = self.fingers_colors[node]
            nodecolor = mcolors.to_rgb(mcolors.CSS4_COLORS[nodecolor])

            for right_idx, right_node in enumerate(right_cm):
                if right_node == node:
                    self.colors[right_idx * 2 + offset, :-1] = nodecolor
                    self.colors[right_idx * 2 + offset + 1, :-1] = nodecolor

    def _transform_mediapipe_to_DHG1428(self, gs_data: np.ndarray) -> np.ndarray:
        gs_data = gs_data.copy()
        gs_data *= [-1, 1, 1]

        mp_finger_bases = [5, 9, 13, 17]
        finger_bases_centers = np.mean(gs_data[:, mp_finger_bases], axis=1)
        thumb_bases = gs_data[:, 0]
        palm_centers = np.mean([finger_bases_centers, thumb_bases], axis=0)

        gs_data = np.insert(gs_data, 1, palm_centers, axis=1)
        return gs_data

    def _interpolate_gesture_sequence(self, gs_data: np.ndarray, target_length=250) -> np.ndarray:
        n_skeletons, n_joints, n_coords = gs_data.shape
        gs = gs_data.reshape((n_skeletons, -1))

        out = []
        for sk in range(np.size(gs, 1)):
            out.append(ndimage.zoom(gs.T[sk], target_length / len(gs), mode="reflect"))

        return np.array(out).T.reshape((-1, n_joints, n_coords))

    def _prepare_sequence(self, sequence: np.ndarray):
        gs_data = np.asarray(sequence, dtype=np.float32)

        gs_length = len(gs_data) * 2
        if self.dhg1428_mode:
            gs_data = self._transform_mediapipe_to_DHG1428(gs_data)

        gs_data = self._interpolate_gesture_sequence(gs_data, gs_length)

        mins = np.min(np.min(np.min(gs_data, axis=2), axis=1), axis=0)
        maxes = np.max(np.max(np.max(gs_data, axis=2), axis=1), axis=0)
        means = np.mean(np.mean(gs_data, axis=1), axis=0)

        gs_data -= means - self.sz_canvas / 2.0
        cam_padding = (-0.5) if self.dhg1428_mode else 0.125
        cam_scaling = (maxes - mins + cam_padding).round(2)

        return gs_data, cam_scaling

    def _render_one_view(
        self,
        gs_data: np.ndarray,
        cam_scaling,
        view_name: str,
        save_path: Path,
    ):
        self._set_view(view_name, cam_scaling)
        tips_coords = {tip: [] for tip in self.finger_tips}

        for idx_skeleton in range(len(gs_data)):
            skeleton_coords = gs_data[idx_skeleton]
            for tip in self.finger_tips:
                if np.sum(skeleton_coords[tip]) != 0:
                    tips_coords[tip].append(skeleton_coords[tip])

        last_skeleton_coords = gs_data[-1]
        coords = []
        for pt1, pt2 in self.connection_map:
            coords.append(last_skeleton_coords[pt1])
            coords.append(last_skeleton_coords[pt2])
        coords = np.array(coords)

        if self.vo_skeletons:
            self.skeleton.set_data(
                coords,
                color=self.colors,
                width=self.w_visuals,
                connect="segments",
            )
            self.skeleton_nodes.set_data(
                coords,
                size=2 * self.w_visuals,
                face_color=self.colors,
                edge_color=self.colors,
            )

        for tip in self.finger_tips:
            if not tips_coords[tip]:
                empty_coords = np.zeros((2, 3), dtype=np.float32)
                empty_colors = np.zeros((2, 4), dtype=np.float32)
                if self.temporal_trails == "lines":
                    self.tips_visuals[tip].set_data(empty_coords, color=empty_colors, width=0)
                else:
                    self.tips_visuals[tip].set_data(empty_coords, face_color=empty_colors, edge_color=empty_colors, size=0)
                continue

            color = self._get_tip_colormap(tip, len(tips_coords[tip]))
            tips_arr = np.array(tips_coords[tip])

            if self.temporal_trails == "lines" and len(tips_arr) == 1:
                tips_arr = np.vstack([tips_arr, tips_arr])
                color = np.vstack([color, color])

            if self.temporal_trails == "lines":
                self.tips_visuals[tip].set_data(tips_arr, color=color, width=self.w_visuals)
            else:
                self.tips_visuals[tip].set_data(tips_arr, face_color=color, edge_color=color, size=self.w_visuals)

        self.canvas.app.process_events()
        img = self.canvas.render()
        vispy.io.write_png(str(save_path), img)

    def render_sequence_to_folder(self, sequence: np.ndarray, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        gs_data, cam_scaling = self._prepare_sequence(sequence)

        for v in MV_ORIENTATIONS:
            self._render_one_view(gs_data, cam_scaling, v, out_dir / f"{v}.png")


_RENDERER = None

def _get_renderer():
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = OfflineFusionRenderer()
    return _RENDERER

# ---------------------------
# INFERENCE
# ---------------------------
def _predict_from_image_folder(folder: Path) -> str:
    data_files = L([folder])
    data_dl = LEARN.dls.test_dl(data_files)
    preds, _, decoded = LEARN.get_preds(dl=data_dl, with_decoded=True)

    if isinstance(preds, (tuple, list, L)):
        preds_np = np.array([i.numpy() for i in preds])
    else:
        preds_np = np.array([preds.numpy()]) 
        
    aggregate = preds_np.sum(axis=0).argmax(axis=1).tolist()

    pred_idx = aggregate[0]
    pred_label = LEARN.dls.vocab[pred_idx]
    return str(pred_label)

def predict_from_sequence(sequence: np.ndarray) -> str:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence shape (T, J, D), got {sequence.shape}")

    tmp_root = Path(tempfile.mkdtemp(prefix="offline_hgr_"))
    try:
        sample_dir = tmp_root / "sample_001"
        renderer = _get_renderer()
        renderer.render_sequence_to_folder(sequence, sample_dir)

        pred_label = _predict_from_image_folder(sample_dir)

        if pred_label not in VOCAB_SET:
            raise RuntimeError(
                f"Predicted label '{pred_label}' not found in learner vocab {VOCAB}"
            )

        return pred_label
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    print("Offline adapter ready.")
    print("Learner vocab:", VOCAB)
    print("Views:", MV_ORIENTATIONS)