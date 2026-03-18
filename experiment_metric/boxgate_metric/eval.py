# # eval.py
# # ------------------------------------------------------------
# # OBJECTIVE 2 - FINAL EVALUATION (SEGMENTATION + INFERENCE)
# #
# # Pipeline:
# #   eval_merge.py -> saves gs_sequence.npy + meta.json
# #   eval.py       -> runs segmentation metrics + gesture inference
# #
# # Metrics:
# #   1. Termination Overhead (frames)      ↓
# #   2. Segmentation Purity (%)            ↑
# #   3. Segment Length Std Dev (frames)    ↓
# #   4. Inference Accuracy (%)             ↑
# #
# # Final outputs:
# #   - Trigger-based table CSV
# #   - Lighting-based table CSV
# #   - Trial-level CSV
# #   - Optional plots
# # ------------------------------------------------------------

# import os
# import json
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from matplotlib.patches import ConnectionPatch


# CONFIG = {
#     "data": {
#         "baseline_dir": "./metric_tests/baseline_manual",
#         "proposed_dir": "./metric_tests/proposed_auto",
#     },
#     "metric1": {
#         "stillness_thresh": 0.01,
#         "inset_last_n": 20,
#         "inset_ylim": (0.0, 0.05),
#         "main_ylim": (0.0, 0.30),
#         "save_plot": True,
#         "plot_path": "./reports/objective2_metric1_termination_overhead.png",
#         "max_left_steps": None,
#     },
#     "metric2": {
#         "action_move_thresh": 0.15,
#         "save_plot": True,
#         "plot_path": "./reports/objective2_metric2_purity_boxplot.png",
#     },
#     "inference": {
#         # If gestureClassInference.py exports:
#         #   predict_from_sequence(sequence: np.ndarray) -> str
#         # this will work automatically.
#         "enabled": True,
#         "write_prediction_back_to_meta": True,
#         "prediction_key": "predicted_label",
#         "truth_key": "gesture_label",
#     },
#     "reports_dir": "./reports",
#     "csv": {
#         "trial_level_path": "./reports/objective2_trial_level_metrics.csv",
#         "trigger_table_path": "./reports/objective2_trigger_table.csv",
#         "lighting_table_path": "./reports/objective2_lighting_table.csv",
#     },
# }


# # ------------------------------------------------------------
# # OPTIONAL INFERENCE ADAPTER
# # ------------------------------------------------------------
# _PREDICT_FN = None
# _INFERENCE_IMPORT_ERROR = None

# try:
#     # You should add this function inside gestureClassInference.py
#     # def predict_from_sequence(sequence: np.ndarray) -> str:
#     #     ...
#     from offline_inference_adapter import predict_from_sequence as _PREDICT_FN
# except Exception as e:
#     import traceback                                  # <--- ADD THIS
#     _INFERENCE_IMPORT_ERROR = traceback.format_exc()  # <--- CHANGE THIS
#     _PREDICT_FN = None


# def infer_label_from_sequence(sequence: np.ndarray) -> str | None:
#     """
#     Runs gesture inference on one saved sequence.

#     Expected:
#         gestureClassInference.py exports:
#             predict_from_sequence(sequence: np.ndarray) -> str

#     Returns:
#         predicted label string, or None if inference is unavailable/fails.
#     """
#     if not CONFIG["inference"]["enabled"]:
#         return None

#     if _PREDICT_FN is None:
#         return None

#     try:
#         pred = _PREDICT_FN(sequence)
#         if pred is None:
#             return None
#         return str(pred)
#     except Exception as e:
#         print(f"[WARN] Inference failed on one trial: {e}")
#         return None


# # ------------------------------------------------------------
# # UTILITIES
# # ------------------------------------------------------------
# def ensure_dir(path: str) -> None:
#     if path and not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)


# def load_json(path: Path) -> dict | None:
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return None


# def save_json(path: Path, obj: dict) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)


# def discover_trial_dirs(root_dir: str) -> list[Path]:
#     root = Path(root_dir)
#     if not root.exists():
#         return []

#     trial_dirs = []
#     for meta_path in root.rglob("meta.json"):
#         trial_dirs.append(meta_path.parent)

#     for npy_path in root.rglob("gs_sequence.npy"):
#         if npy_path.parent not in trial_dirs:
#             trial_dirs.append(npy_path.parent)

#     return sorted(set(trial_dirs))


# def infer_method_from_path(path: Path) -> str:
#     p = str(path).replace("\\", "/")
#     if "/baseline_manual/" in p:
#         return "baseline_manual"
#     if "/proposed_auto/" in p:
#         return "proposed_auto"
#     return "unknown"


# def infer_scenario_from_path(path: Path, method_root_name: str) -> str:
#     parts = path.parts
#     if method_root_name in parts:
#         idx = parts.index(method_root_name)
#         if idx + 1 < len(parts):
#             return parts[idx + 1]
#     return "unknown"


# def load_trial(trial_dir: Path) -> dict | None:
#     npy_path = trial_dir / "gs_sequence.npy"
#     meta_path = trial_dir / "meta.json"

#     meta = load_json(meta_path) if meta_path.exists() else {}

#     arr = None
#     if npy_path.exists():
#         try:
#             arr = np.load(str(npy_path))
#         except Exception:
#             arr = None

#     method = meta.get("method")
#     if method is None:
#         method = infer_method_from_path(trial_dir)

#     method_root_name = "baseline_manual" if method == "baseline_manual" else "proposed_auto"
#     scenario = meta.get("scenario")
#     if scenario is None:
#         scenario = infer_scenario_from_path(trial_dir, method_root_name)

#     return {
#         "trial_dir": str(trial_dir),
#         "trial_id": meta.get("trial_id", trial_dir.name),
#         "method": method,
#         "scenario": scenario,
#         "meta_path": str(meta_path),
#         "meta": meta,
#         "gs": arr,
#     }


# def load_all_trials(baseline_root: str, proposed_root: str) -> list[dict]:
#     trials = []
#     for trial_dir in discover_trial_dirs(baseline_root):
#         t = load_trial(trial_dir)
#         if t is not None:
#             trials.append(t)
#     for trial_dir in discover_trial_dirs(proposed_root):
#         t = load_trial(trial_dir)
#         if t is not None:
#             trials.append(t)
#     return trials


# # ------------------------------------------------------------
# # SEQUENCE PROCESSING
# # ------------------------------------------------------------
# def load_wrist_positions_from_array(data: np.ndarray, wrist_idx: int = 0) -> np.ndarray | None:
#     if data is None:
#         return None
#     if not isinstance(data, np.ndarray) or data.ndim != 3:
#         return None
#     if data.shape[1] <= wrist_idx or data.shape[2] < 2:
#         return None
#     return data[:, wrist_idx, :3]


# def step_displacement(pos: np.ndarray) -> np.ndarray:
#     if pos is None or len(pos) < 2:
#         return np.array([])
#     return np.linalg.norm(pos[1:] - pos[:-1], axis=1)


# def velocity_sequence_from_trial(trial: dict, wrist_idx: int = 0) -> np.ndarray | None:
#     gs = trial.get("gs")
#     wrist = load_wrist_positions_from_array(gs, wrist_idx=wrist_idx)
#     if wrist is None or len(wrist) < 2:
#         return None
#     v = step_displacement(wrist)
#     return v if v.size else None


# def segment_length_frames(trial: dict) -> int:
#     gs = trial.get("gs")
#     if gs is None or not isinstance(gs, np.ndarray):
#         return 0
#     return int(len(gs))


# # def purity_percent_from_trial(trial: dict, action_move_thresh: float) -> float | None:
# #     v = velocity_sequence_from_trial(trial, wrist_idx=0)
# #     if v is None or len(v) == 0:
# #         return None
# #     return float((np.sum(v > action_move_thresh) / len(v)) * 100.0)

# def purity_percent_from_trial(trial: dict, relative_thresh: float) -> float | None:
#     v = velocity_sequence_from_trial(trial, wrist_idx=0)
#     if v is None or len(v) == 0:
#         return None
        
#     # 1. Find the peak speed of this specific gesture trial
#     peak_speed = float(np.max(v))
    
#     # Safety catch for completely dead sensors
#     if peak_speed == 0:
#         return 0.0
        
#     # 2. Define the dynamic threshold (e.g., 15% of the peak speed)
#     dynamic_thresh = peak_speed * relative_thresh
    
#     # 3. Count how many frames are part of the high-speed "core" action
#     core_frames = np.sum(v > dynamic_thresh)
    
#     # 4. Calculate true density
#     return float((core_frames / len(v)) * 100.0)


# def wasted_tail_frames_from_trial(trial: dict, still_thresh: float) -> int | None:
#     v = velocity_sequence_from_trial(trial, wrist_idx=0)
#     if v is None or v.size == 0:
#         return None

#     idx = np.where(v > still_thresh)[0]
#     if idx.size == 0:
#         wasted = int(v.size)
#     else:
#         last_active = int(idx[-1])
#         wasted = int(v.size - 1 - last_active)
#     return wasted


# def build_end_aligned_matrix_from_trials(trials: list[dict]) -> tuple[np.ndarray, int] | tuple[None, None]:
#     seqs = []
#     for t in trials:
#         v = velocity_sequence_from_trial(t, wrist_idx=0)
#         if v is not None:
#             seqs.append(v)

#     if not seqs:
#         return None, None

#     max_len = max(len(v) for v in seqs)
#     mat = np.full((len(seqs), max_len), np.nan, dtype=np.float32)

#     for i, v in enumerate(seqs):
#         mat[i, -len(v):] = v

#     return mat, max_len


# def left_pad_to(mat: np.ndarray, target_len: int) -> np.ndarray:
#     if mat.shape[1] == target_len:
#         return mat
#     pad = target_len - mat.shape[1]
#     return np.pad(mat, ((0, 0), (pad, 0)), constant_values=np.nan)


# def nanmean_nanstd(mat: np.ndarray):
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
#         warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
#         mean = np.nanmean(mat, axis=0)
#         std = np.nanstd(mat, axis=0)

#     n = np.sum(~np.isnan(mat), axis=0)
#     return mean, std, n


# # ------------------------------------------------------------
# # OPTIONAL PLOTS
# # ------------------------------------------------------------
# def run_metric1_plot(baseline_trials: list[dict], proposed_trials: list[dict], cfg: dict) -> dict:
#     still_thresh = float(cfg["stillness_thresh"])
#     inset_last_n = int(cfg["inset_last_n"])
#     inset_ylim = tuple(cfg["inset_ylim"])
#     main_ylim = tuple(cfg["main_ylim"])
#     max_left_steps = cfg.get("max_left_steps", None)

#     base_mat, _ = build_end_aligned_matrix_from_trials(baseline_trials)
#     prop_mat, _ = build_end_aligned_matrix_from_trials(proposed_trials)

#     if base_mat is None or prop_mat is None:
#         return {"ok": False, "error": "Metric 1 plot: no valid trial sequences found."}

#     max_len = max(base_mat.shape[1], prop_mat.shape[1])
#     base_mat = left_pad_to(base_mat, max_len)
#     prop_mat = left_pad_to(prop_mat, max_len)

#     base_mean, base_std, _ = nanmean_nanstd(base_mat)
#     prop_mean, prop_std, _ = nanmean_nanstd(prop_mat)

#     x = np.arange(-(max_len - 1), 1)

#     if isinstance(max_left_steps, int) and 0 < max_left_steps < max_len:
#         take = max_left_steps + 1
#         x = x[-take:]
#         base_mean, base_std = base_mean[-take:], base_std[-take:]
#         prop_mean, prop_std = prop_mean[-take:], prop_std[-take:]

#     base_wasted = [wasted_tail_frames_from_trial(t, still_thresh) for t in baseline_trials]
#     base_wasted = [x for x in base_wasted if x is not None]
#     prop_wasted = [wasted_tail_frames_from_trial(t, still_thresh) for t in proposed_trials]
#     prop_wasted = [x for x in prop_wasted if x is not None]

#     base_wasted_total = int(np.sum(base_wasted)) if base_wasted else 0
#     prop_wasted_total = int(np.sum(prop_wasted)) if prop_wasted else 0

#     fig, ax = plt.subplots(figsize=(12, 7))

#     ax.plot(x, base_mean, linewidth=2, label="Baseline (Manual Stop)")
#     ax.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.15)

#     ax.plot(x, prop_mean, linewidth=2, label="Proposed (BoxGate Stop)")
#     ax.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.15)

#     ax.axhline(
#         y=still_thresh,
#         linestyle=":",
#         linewidth=1.5,
#         color="red",
#         label=f"Stillness Threshold ({still_thresh:g})"
#     )

#     ax.set_title("Metric 1: Termination Overhead", fontsize=14, pad=16, fontweight="bold")
#     ax.set_ylabel("Wrist Motion Magnitude (Inter-frame Displacement)", fontsize=12)
#     ax.set_xlabel("Frames Before Stop", fontsize=12)
#     ax.set_xlim(x[0], 0)
#     ax.set_ylim(main_ylim[0], main_ylim[1])
#     ax.grid(True, alpha=0.35)
#     leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95)

#     txt = (
#         "Termination overhead summary\n"
#         f"Stillness threshold: {still_thresh:g}\n"
#         f"Baseline: {base_wasted_total:,} frames\n"
#         f"Proposed: {prop_wasted_total:,} frames"
#     )

#     fig.canvas.draw()
#     renderer = fig.canvas.get_renderer()
#     bbox_disp = leg.get_window_extent(renderer=renderer)
#     (x1, y0) = ax.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y0))

#     ax.text(
#         x1, y0 - 0.01, txt,
#         transform=ax.transAxes,
#         fontsize=10,
#         va="top", ha="right",
#         zorder=20,
#         bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75", boxstyle="round,pad=0.4")
#     )

#     axins = ax.inset_axes([0.24, 0.57, 0.52, 0.26])

#     axins.plot(x, base_mean, linewidth=2)
#     axins.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.20)

#     axins.plot(x, prop_mean, linewidth=2)
#     axins.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.20)

#     axins.axhline(y=still_thresh, linestyle=":", linewidth=1.5, color="red")
#     axins.axhspan(inset_ylim[0], still_thresh, color="red", alpha=0.18, zorder=0)

#     axins.set_xlim(-inset_last_n, 0)
#     axins.set_ylim(inset_ylim[0], inset_ylim[1])
#     axins.set_xticks([])
#     axins.set_yticks([])
#     axins.set_title(f"Last {inset_last_n} Frames", fontsize=10)
#     axins.patch.set_facecolor("white")
#     axins.patch.set_alpha(0.90)
#     axins.set_zorder(10)

#     pp, p1, p2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)
#     p1.set_visible(False)
#     p2.set_visible(False)

#     con = ConnectionPatch(
#         xyA=(1, 0), coordsA=axins.transAxes,
#         xyB=(-inset_last_n / 2, inset_ylim[1]), coordsB=ax.transData,
#         color="0.5", lw=1
#     )
#     ax.add_artist(con)

#     plt.tight_layout()
#     if cfg.get("save_plot", False):
#         ensure_dir(os.path.dirname(cfg["plot_path"]))
#         plt.savefig(cfg["plot_path"], dpi=200)
#     plt.show()

#     return {
#         "ok": True,
#         "baseline_trials": len(baseline_trials),
#         "proposed_trials": len(proposed_trials),
#         "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
#     }


# def run_metric2_plot(baseline_trials: list[dict], proposed_trials: list[dict], cfg: dict) -> dict:
#     thresh = float(cfg["action_move_thresh"])

#     base_vals = []
#     for t in baseline_trials:
#         v = purity_percent_from_trial(t, thresh)
#         if v is not None:
#             base_vals.append(v)

#     prop_vals = []
#     for t in proposed_trials:
#         v = purity_percent_from_trial(t, thresh)
#         if v is not None:
#             prop_vals.append(v)

#     if len(base_vals) == 0 or len(prop_vals) == 0:
#         return {"ok": False, "error": "Metric 2 plot: no valid purity values found."}

#     base_mean, base_std = float(np.mean(base_vals)), float(np.std(base_vals))
#     prop_mean, prop_std = float(np.mean(prop_vals)), float(np.std(prop_vals))

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.boxplot([base_vals, prop_vals], tick_labels=["Baseline", "Proposed"], showmeans=True)

#     rng = np.random.default_rng(42)
#     ax.scatter(1 + rng.normal(0, 0.03, size=len(base_vals)), base_vals, alpha=0.5, s=18)
#     ax.scatter(2 + rng.normal(0, 0.03, size=len(prop_vals)), prop_vals, alpha=0.5, s=18)

#     ax.set_title("Metric 2: Segmentation Purity", fontsize=13, pad=12, fontweight="bold")
#     ax.set_ylabel("Purity (%) = ActionSteps / TotalSteps × 100", fontsize=11)
#     ax.grid(True, axis="y", alpha=0.3)

#     txt = (
#         "Segmentation purity summary\n"
#         f"Action threshold: {thresh:g}\n"
#         f"Baseline: {base_mean:.2f}% ± {base_std:.2f}% (n={len(base_vals)})\n"
#         f"Proposed: {prop_mean:.2f}% ± {prop_std:.2f}% (n={len(prop_vals)})"
#     )

#     ax.text(
#         0.98, 0.98, txt,
#         transform=ax.transAxes,
#         fontsize=10,
#         va="top", ha="right",
#         bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75", boxstyle="round,pad=0.4")
#     )

#     plt.tight_layout()
#     if cfg.get("save_plot", False):
#         ensure_dir(os.path.dirname(cfg["plot_path"]))
#         plt.savefig(cfg["plot_path"], dpi=200)
#     plt.show()

#     return {
#         "ok": True,
#         "baseline_n_valid": len(base_vals),
#         "proposed_n_valid": len(prop_vals),
#         "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
#     }


# # ------------------------------------------------------------
# # TRIAL-LEVEL METRICS
# # ------------------------------------------------------------
# def maybe_write_prediction_back(meta_path: str, meta: dict, prediction: str | None) -> None:
#     if not CONFIG["inference"]["write_prediction_back_to_meta"]:
#         return
#     if prediction is None:
#         return
#     meta[CONFIG["inference"]["prediction_key"]] = prediction
#     save_json(Path(meta_path), meta)


# def compute_trial_level_metrics(trials: list[dict], cfg: dict) -> pd.DataFrame:
#     still_thresh = float(cfg["metric1"]["stillness_thresh"])
#     purity_thresh = float(cfg["metric2"]["action_move_thresh"])

#     rows = []
#     for t in trials:
#         meta = t.get("meta", {})
#         gs = t.get("gs")

#         n_frames = segment_length_frames(t)
#         wasted = wasted_tail_frames_from_trial(t, still_thresh)
#         purity = purity_percent_from_trial(t, purity_thresh)

#         gesture_label = meta.get(CONFIG["inference"]["truth_key"])

#         predicted_label = meta.get(CONFIG["inference"]["prediction_key"])
#         if predicted_label is None and gs is not None:
#             predicted_label = infer_label_from_sequence(gs)
#             maybe_write_prediction_back(t["meta_path"], meta, predicted_label)

#         inference_correct = None
#         if gesture_label is not None and predicted_label is not None:
#             inference_correct = int(str(gesture_label) == str(predicted_label))

#         rows.append({
#             "trial_id": t["trial_id"],
#             "trial_dir": t["trial_dir"],
#             "method": t["method"],
#             "scenario": meta.get("scenario", t["scenario"]),
#             "lighting": meta.get("lighting", "unknown"),
#             "group_type": meta.get("group_type", "unknown"),
#             "gesture_label": gesture_label,
#             "trigger_family": meta.get("trigger_family", "unknown"),
#             "trial_index": meta.get("trial_index"),
#             "round_index": meta.get("round_index"),
#             "valid_capture": meta.get("valid_capture", True),
#             "n_frames": n_frames,
#             "termination_overhead_frames": wasted,
#             "segmentation_purity_percent": purity,
#             "predicted_label": predicted_label,
#             "inference_correct": inference_correct,
#         })

#     return pd.DataFrame(rows)


# # ------------------------------------------------------------
# # FINAL TABLES
# # ------------------------------------------------------------
# def build_trigger_table(df: pd.DataFrame) -> pd.DataFrame:
#     work = df[df["valid_capture"] == True].copy()
#     if work.empty:
#         return pd.DataFrame()

#     grouped = work.groupby(["trigger_family", "method"], dropna=False)

#     summary = grouped.agg(
#         n_valid_trials=("trial_id", "count"),
#         termination_overhead_frames=("termination_overhead_frames", "mean"),
#         segmentation_purity_percent=("segmentation_purity_percent", "mean"),
#         segment_length_std_dev=("n_frames", "std"),
#         inference_accuracy=("inference_correct", "mean"),
#     ).reset_index()

#     summary["inference_accuracy"] = summary["inference_accuracy"] * 100.0
#     summary = summary.rename(columns={"trigger_family": "trigger_scenario"})
#     return summary


# def build_lighting_table(df: pd.DataFrame) -> pd.DataFrame:
#     work = df[(df["valid_capture"] == True) & (df["group_type"] == "lighting")].copy()
#     if work.empty:
#         return pd.DataFrame()

#     grouped = work.groupby(["lighting", "method"], dropna=False)

#     summary = grouped.agg(
#         n_valid_trials=("trial_id", "count"),
#         termination_overhead_frames=("termination_overhead_frames", "mean"),
#         segmentation_purity_percent=("segmentation_purity_percent", "mean"),
#         segment_length_std_dev=("n_frames", "std"),
#         inference_accuracy=("inference_correct", "mean"),
#     ).reset_index()

#     summary["inference_accuracy"] = summary["inference_accuracy"] * 100.0
#     return summary


# # ------------------------------------------------------------
# # MAIN
# # ------------------------------------------------------------
# def main():
#     ensure_dir(CONFIG["reports_dir"])

#     trials = load_all_trials(CONFIG["data"]["baseline_dir"], CONFIG["data"]["proposed_dir"])
#     if len(trials) == 0:
#         print("No trials found. Run eval_merge.py first.")
#         return

#     baseline_trials = [t for t in trials if t["method"] == "baseline_manual" and t.get("gs") is not None]
#     proposed_trials = [t for t in trials if t["method"] == "proposed_auto" and t.get("gs") is not None]

#     print(f"Loaded trials: total={len(trials)}, baseline={len(baseline_trials)}, proposed={len(proposed_trials)}")

#     if _PREDICT_FN is None:
#         print("\n[INFO] Inference function not found.")
#         print("[INFO] Segmentation metrics will still be computed.")
#         print("[INFO] Inference accuracy will be NaN until you add:")
#         print("       from gestureClassInference import predict_from_sequence")
#         if _INFERENCE_IMPORT_ERROR:
#             print(f"[INFO] Import error was: {_INFERENCE_IMPORT_ERROR}")

#     # Optional plots
#     m1 = run_metric1_plot(baseline_trials, proposed_trials, CONFIG["metric1"])
#     if not m1["ok"]:
#         print(m1["error"])

#     m2 = run_metric2_plot(baseline_trials, proposed_trials, CONFIG["metric2"])
#     if not m2["ok"]:
#         print(m2["error"])

#     # Trial-level metrics + inference
#     df_trials = compute_trial_level_metrics(trials, CONFIG)
#     ensure_dir(os.path.dirname(CONFIG["csv"]["trial_level_path"]))
#     df_trials.to_csv(CONFIG["csv"]["trial_level_path"], index=False)

#     # Final tables
#     df_trigger = build_trigger_table(df_trials)
#     df_lighting = build_lighting_table(df_trials)

#     if not df_trigger.empty:
#         df_trigger.to_csv(CONFIG["csv"]["trigger_table_path"], index=False)
#         print("\n=== OBJECTIVE 2: TRIGGER-BASED RESULTS TABLE ===")
#         print(df_trigger.to_string(index=False))
#     else:
#         print("\nNo valid trigger-based table could be generated.")

#     if not df_lighting.empty:
#         df_lighting.to_csv(CONFIG["csv"]["lighting_table_path"], index=False)
#         print("\n=== OBJECTIVE 2: LIGHTING RESULTS TABLE ===")
#         print(df_lighting.to_string(index=False))
#     else:
#         print("\nNo valid lighting table could be generated.")

#     print("\nSaved reports:")
#     print(f" - Trial-level CSV     : {CONFIG['csv']['trial_level_path']}")
#     print(f" - Trigger table CSV   : {CONFIG['csv']['trigger_table_path']}")
#     print(f" - Lighting table CSV  : {CONFIG['csv']['lighting_table_path']}")
#     if m1.get("plot_path"):
#         print(f" - Metric 1 plot       : {m1['plot_path']}")
#     if m2.get("plot_path"):
#         print(f" - Metric 2 plot       : {m2['plot_path']}")

#     print("\nFinal paper tables expected:")
#     print("Table 1: Trigger-Based Results")
#     print("| Trigger Scenario | Method | Termination Overhead (frames) ↓ | Segmentation Purity (%) ↑ | Segment Length Std Dev ↓ | Inference Accuracy (%) ↑ |")
#     print("\nTable 2: Lighting Results")
#     print("| Lighting Condition | Method | Termination Overhead (frames) ↓ | Segmentation Purity (%) ↑ | Segment Length Std Dev ↓ | Inference Accuracy (%) ↑ |")


# if __name__ == "__main__":
#     main()

# eval.py
# ------------------------------------------------------------
# OBJECTIVE 2 - FINAL EVALUATION (SEGMENTATION + INFERENCE)
#
# Pipeline:
#   eval_merge.py -> saves gs_sequence.npy + meta.json
#   eval.py       -> runs segmentation metrics + gesture inference
#
# Metrics:
#   1. Termination Overhead (frames)      ↓
#   2. Segmentation Purity (%)            ↑
#   3. Segment Length Std Dev (frames)    ↓
#   4. Inference Accuracy (%)             ↑
#
# Final outputs:
#   - Trigger-based table CSV
#   - Lighting-based table CSV
#   - Trial-level CSV
#   - Optional plots
# ------------------------------------------------------------

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch


CONFIG = {
    "data": {
        "baseline_dir": "./metric_tests/baseline_manual",
        "proposed_dir": "./metric_tests/proposed_auto",
    },
    "metric1": {
        "stillness_thresh": 0.001,
        "inset_last_n": 20,
        "inset_ylim": (0.0, 0.05),
        "main_ylim": (0.0, 0.30),
        "save_plot": True,
        "plot_path": "./reports/objective2_metric1_termination_overhead.png",
        "max_left_steps": None,
    },
    "metric2": {
        "action_move_thresh": 0.15,
        "save_plot": True,
        "plot_path": "./reports/objective2_metric2_purity_boxplot.png",
    },
    "inference": {
        "enabled": True,
        "write_prediction_back_to_meta": True,
        "prediction_key": "predicted_label",
        "truth_key": "gesture_label",
    },
    "reports_dir": "./reports",
    "csv": {
        "trial_level_path": "./reports/objective2_trial_level_metrics.csv",
        "trigger_table_path": "./reports/objective2_trigger_table.csv",
        "lighting_table_path": "./reports/objective2_lighting_table.csv",
    },
}


# ------------------------------------------------------------
# OPTIONAL INFERENCE ADAPTER
# ------------------------------------------------------------
_PREDICT_FN = None
_INFERENCE_IMPORT_ERROR = None

try:
    from offline_inference_adapter import predict_from_sequence as _PREDICT_FN
except Exception as e:
    import traceback                                  
    _INFERENCE_IMPORT_ERROR = traceback.format_exc()  
    _PREDICT_FN = None


def infer_label_from_sequence(sequence: np.ndarray) -> str | None:
    if not CONFIG["inference"]["enabled"]:
        return None

    if _PREDICT_FN is None:
        return None

    try:
        pred = _PREDICT_FN(sequence)
        if pred is None:
            return None
        return str(pred)
    except Exception as e:
        print(f"[WARN] Inference failed on one trial: {e}")
        return None


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def discover_trial_dirs(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    trial_dirs = []
    for meta_path in root.rglob("meta.json"):
        trial_dirs.append(meta_path.parent)

    for npy_path in root.rglob("gs_sequence.npy"):
        if npy_path.parent not in trial_dirs:
            trial_dirs.append(npy_path.parent)

    return sorted(set(trial_dirs))


def infer_method_from_path(path: Path) -> str:
    p = str(path).replace("\\", "/")
    if "/baseline_manual/" in p:
        return "baseline_manual"
    if "/proposed_auto/" in p:
        return "proposed_auto"
    return "unknown"


def infer_scenario_from_path(path: Path, method_root_name: str) -> str:
    parts = path.parts
    if method_root_name in parts:
        idx = parts.index(method_root_name)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def load_trial(trial_dir: Path) -> dict | None:
    npy_path = trial_dir / "gs_sequence.npy"
    meta_path = trial_dir / "meta.json"

    meta = load_json(meta_path) if meta_path.exists() else {}

    arr = None
    if npy_path.exists():
        try:
            arr = np.load(str(npy_path))
        except Exception:
            arr = None

    method = meta.get("method")
    if method is None:
        method = infer_method_from_path(trial_dir)

    method_root_name = "baseline_manual" if method == "baseline_manual" else "proposed_auto"
    scenario = meta.get("scenario")
    if scenario is None:
        scenario = infer_scenario_from_path(trial_dir, method_root_name)

    return {
        "trial_dir": str(trial_dir),
        "trial_id": meta.get("trial_id", trial_dir.name),
        "method": method,
        "scenario": scenario,
        "meta_path": str(meta_path),
        "meta": meta,
        "gs": arr,
    }


def load_all_trials(baseline_root: str, proposed_root: str) -> list[dict]:
    trials = []
    for trial_dir in discover_trial_dirs(baseline_root):
        t = load_trial(trial_dir)
        if t is not None:
            trials.append(t)
    for trial_dir in discover_trial_dirs(proposed_root):
        t = load_trial(trial_dir)
        if t is not None:
            trials.append(t)
    return trials


# ------------------------------------------------------------
# SEQUENCE PROCESSING
# ------------------------------------------------------------
def load_wrist_positions_from_array(data: np.ndarray, wrist_idx: int = 0) -> np.ndarray | None:
    if data is None:
        return None
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        return None
    if data.shape[1] <= wrist_idx or data.shape[2] < 2:
        return None
    return data[:, wrist_idx, :3]


def step_displacement(pos: np.ndarray) -> np.ndarray:
    if pos is None or len(pos) < 2:
        return np.array([])
    return np.linalg.norm(pos[1:] - pos[:-1], axis=1)


def velocity_sequence_from_trial(trial: dict, wrist_idx: int = 0) -> np.ndarray | None:
    gs = trial.get("gs")
    wrist = load_wrist_positions_from_array(gs, wrist_idx=wrist_idx)
    if wrist is None or len(wrist) < 2:
        return None
    v = step_displacement(wrist)
    return v if v.size else None


def segment_length_frames(trial: dict) -> int:
    gs = trial.get("gs")
    if gs is None or not isinstance(gs, np.ndarray):
        return 0
    return int(len(gs))


def purity_percent_from_trial(trial: dict, relative_thresh: float) -> float | None:
    v = velocity_sequence_from_trial(trial, wrist_idx=0)
    if v is None or len(v) == 0:
        return None
        
    # 1. Find the peak speed of this specific gesture trial
    peak_speed = float(np.max(v))
    
    # Safety catch for completely dead sensors
    if peak_speed == 0:
        return 0.0
        
    # 2. Define the dynamic threshold (e.g., 15% of the peak speed)
    dynamic_thresh = peak_speed * relative_thresh
    
    # 3. Count how many frames are part of the high-speed "core" action
    core_frames = np.sum(v > dynamic_thresh)
    
    # 4. Calculate true density
    return float((core_frames / len(v)) * 100.0)


def wasted_tail_frames_from_trial(trial: dict, still_thresh: float) -> int | None:
    v = velocity_sequence_from_trial(trial, wrist_idx=0)
    if v is None or v.size == 0:
        return None

    idx = np.where(v > still_thresh)[0]
    if idx.size == 0:
        wasted = int(v.size)
    else:
        last_active = int(idx[-1])
        wasted = int(v.size - 1 - last_active)
    return wasted


def build_end_aligned_matrix_from_trials(trials: list[dict]) -> tuple[np.ndarray, int] | tuple[None, None]:
    seqs = []
    for t in trials:
        v = velocity_sequence_from_trial(t, wrist_idx=0)
        if v is not None:
            seqs.append(v)

    if not seqs:
        return None, None

    max_len = max(len(v) for v in seqs)
    mat = np.full((len(seqs), max_len), np.nan, dtype=np.float32)

    for i, v in enumerate(seqs):
        mat[i, -len(v):] = v

    return mat, max_len


def left_pad_to(mat: np.ndarray, target_len: int) -> np.ndarray:
    if mat.shape[1] == target_len:
        return mat
    pad = target_len - mat.shape[1]
    return np.pad(mat, ((0, 0), (pad, 0)), constant_values=np.nan)


def nanmean_nanstd(mat: np.ndarray):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)

    n = np.sum(~np.isnan(mat), axis=0)
    return mean, std, n


# ------------------------------------------------------------
# OPTIONAL PLOTS
# ------------------------------------------------------------
def run_metric1_plot(baseline_trials: list[dict], proposed_trials: list[dict], cfg: dict) -> dict:
    still_thresh = float(cfg["stillness_thresh"])
    inset_last_n = int(cfg["inset_last_n"])
    inset_ylim = tuple(cfg["inset_ylim"])
    main_ylim = tuple(cfg["main_ylim"])
    max_left_steps = cfg.get("max_left_steps", None)

    base_mat, _ = build_end_aligned_matrix_from_trials(baseline_trials)
    prop_mat, _ = build_end_aligned_matrix_from_trials(proposed_trials)

    if base_mat is None or prop_mat is None:
        return {"ok": False, "error": "Metric 1 plot: no valid trial sequences found."}

    max_len = max(base_mat.shape[1], prop_mat.shape[1])
    base_mat = left_pad_to(base_mat, max_len)
    prop_mat = left_pad_to(prop_mat, max_len)

    base_mean, base_std, _ = nanmean_nanstd(base_mat)
    prop_mean, prop_std, _ = nanmean_nanstd(prop_mat)

    x = np.arange(-(max_len - 1), 1)

    if isinstance(max_left_steps, int) and 0 < max_left_steps < max_len:
        take = max_left_steps + 1
        x = x[-take:]
        base_mean, base_std = base_mean[-take:], base_std[-take:]
        prop_mean, prop_std = prop_mean[-take:], prop_std[-take:]

    base_wasted = [wasted_tail_frames_from_trial(t, still_thresh) for t in baseline_trials]
    base_wasted = [x for x in base_wasted if x is not None]
    prop_wasted = [wasted_tail_frames_from_trial(t, still_thresh) for t in proposed_trials]
    prop_wasted = [x for x in prop_wasted if x is not None]

    base_wasted_total = int(np.sum(base_wasted)) if base_wasted else 0
    prop_wasted_total = int(np.sum(prop_wasted)) if prop_wasted else 0

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x, base_mean, linewidth=2, label="Baseline (Manual Stop)")
    ax.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.15)

    ax.plot(x, prop_mean, linewidth=2, label="Proposed (BoxGate Stop)")
    ax.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.15)

    ax.axhline(
        y=still_thresh,
        linestyle=":",
        linewidth=1.5,
        color="red",
        label=f"Stillness Threshold ({still_thresh:g})"
    )

    ax.set_title("Metric 1: Termination Overhead", fontsize=14, pad=16, fontweight="bold")
    ax.set_ylabel("Wrist Motion Magnitude (Inter-frame Displacement)", fontsize=12)
    ax.set_xlabel("Frames Before Stop", fontsize=12)
    ax.set_xlim(x[0], 0)
    ax.set_ylim(main_ylim[0], main_ylim[1])
    ax.grid(True, alpha=0.35)
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    txt = (
        "Termination overhead summary\n"
        f"Stillness threshold: {still_thresh:g}\n"
        f"Baseline: {base_wasted_total:,} frames\n"
        f"Proposed: {prop_wasted_total:,} frames"
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = leg.get_window_extent(renderer=renderer)
    (x1, y0) = ax.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y0))

    ax.text(
        x1, y0 - 0.01, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top", ha="right",
        zorder=20,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75", boxstyle="round,pad=0.4")
    )

    axins = ax.inset_axes([0.24, 0.57, 0.52, 0.26])

    axins.plot(x, base_mean, linewidth=2)
    axins.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.20)

    axins.plot(x, prop_mean, linewidth=2)
    axins.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.20)

    axins.axhline(y=still_thresh, linestyle=":", linewidth=1.5, color="red")
    axins.axhspan(inset_ylim[0], still_thresh, color="red", alpha=0.18, zorder=0)

    axins.set_xlim(-inset_last_n, 0)
    axins.set_ylim(inset_ylim[0], inset_ylim[1])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(f"Last {inset_last_n} Frames", fontsize=10)
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(0.90)
    axins.set_zorder(10)

    pp, p1, p2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)
    p1.set_visible(False)
    p2.set_visible(False)

    con = ConnectionPatch(
        xyA=(1, 0), coordsA=axins.transAxes,
        xyB=(-inset_last_n / 2, inset_ylim[1]), coordsB=ax.transData,
        color="0.5", lw=1
    )
    ax.add_artist(con)

    plt.tight_layout()
    if cfg.get("save_plot", False):
        ensure_dir(os.path.dirname(cfg["plot_path"]))
        plt.savefig(cfg["plot_path"], dpi=200)
    plt.show()

    return {
        "ok": True,
        "baseline_trials": len(baseline_trials),
        "proposed_trials": len(proposed_trials),
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }


def run_metric2_plot(baseline_trials: list[dict], proposed_trials: list[dict], cfg: dict) -> dict:
    thresh = float(cfg["action_move_thresh"])

    base_vals = []
    for t in baseline_trials:
        v = purity_percent_from_trial(t, thresh)
        if v is not None:
            base_vals.append(v)

    prop_vals = []
    for t in proposed_trials:
        v = purity_percent_from_trial(t, thresh)
        if v is not None:
            prop_vals.append(v)

    if len(base_vals) == 0 or len(prop_vals) == 0:
        return {"ok": False, "error": "Metric 2 plot: no valid purity values found."}

    base_mean, base_std = float(np.mean(base_vals)), float(np.std(base_vals))
    prop_mean, prop_std = float(np.mean(prop_vals)), float(np.std(prop_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([base_vals, prop_vals], tick_labels=["Baseline", "Proposed"], showmeans=True)

    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.normal(0, 0.03, size=len(base_vals)), base_vals, alpha=0.5, s=18)
    ax.scatter(2 + rng.normal(0, 0.03, size=len(prop_vals)), prop_vals, alpha=0.5, s=18)

    ax.set_title("Metric 2: Segmentation Purity", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Purity (%) = ActionSteps / TotalSteps × 100", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    txt = (
        "Segmentation purity summary\n"
        f"Action threshold: {thresh:g}\n"
        f"Baseline: {base_mean:.2f}% ± {base_std:.2f}% (n={len(base_vals)})\n"
        f"Proposed: {prop_mean:.2f}% ± {prop_std:.2f}% (n={len(prop_vals)})"
    )

    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75", boxstyle="round,pad=0.4")
    )

    plt.tight_layout()
    if cfg.get("save_plot", False):
        ensure_dir(os.path.dirname(cfg["plot_path"]))
        plt.savefig(cfg["plot_path"], dpi=200)
    plt.show()

    return {
        "ok": True,
        "baseline_n_valid": len(base_vals),
        "proposed_n_valid": len(prop_vals),
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }


# ------------------------------------------------------------
# TRIAL-LEVEL METRICS
# ------------------------------------------------------------
def maybe_write_prediction_back(meta_path: str, meta: dict, prediction: str | None) -> None:
    if not CONFIG["inference"]["write_prediction_back_to_meta"]:
        return
    if prediction is None:
        return
    meta[CONFIG["inference"]["prediction_key"]] = prediction
    save_json(Path(meta_path), meta)


def compute_trial_level_metrics(trials: list[dict], cfg: dict) -> pd.DataFrame:
    still_thresh = float(cfg["metric1"]["stillness_thresh"])
    purity_thresh = float(cfg["metric2"]["action_move_thresh"])

    rows = []
    for t in trials:
        meta = t.get("meta", {})
        gs = t.get("gs")

        n_frames = segment_length_frames(t)
        wasted = wasted_tail_frames_from_trial(t, still_thresh)
        purity = purity_percent_from_trial(t, purity_thresh)

        gesture_label = meta.get(CONFIG["inference"]["truth_key"])

        predicted_label = meta.get(CONFIG["inference"]["prediction_key"])
        if predicted_label is None and gs is not None:
            predicted_label = infer_label_from_sequence(gs)
            maybe_write_prediction_back(t["meta_path"], meta, predicted_label)

        inference_correct = None
        if gesture_label is not None and predicted_label is not None:
            inference_correct = int(str(gesture_label) == str(predicted_label))

        rows.append({
            "trial_id": t["trial_id"],
            "trial_dir": t["trial_dir"],
            "method": t["method"],
            "scenario": meta.get("scenario", t["scenario"]),
            "lighting": meta.get("lighting", "unknown"),
            "group_type": meta.get("group_type", "unknown"),
            "gesture_label": gesture_label,
            "trigger_family": meta.get("trigger_family", "unknown"),
            "trial_index": meta.get("trial_index"),
            "round_index": meta.get("round_index"),
            "valid_capture": meta.get("valid_capture", True),
            "n_frames": n_frames,
            "termination_overhead_frames": wasted,
            "segmentation_purity_percent": purity,
            "predicted_label": predicted_label,
            "inference_correct": inference_correct,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# FINAL TABLES
# ------------------------------------------------------------
def build_trigger_table(df: pd.DataFrame) -> pd.DataFrame:
    # ONLY explicitly allow trigger scenarios into this table
    target_scenarios = ["wrist_trigger", "scale_trigger"]
    work = df[(df["valid_capture"] == True) & (df["scenario"].isin(target_scenarios))].copy()
    
    if work.empty:
        return pd.DataFrame()

    # Group by scenario instead of inferred trigger_family
    grouped = work.groupby(["scenario", "method"], dropna=False)

    summary = grouped.agg(
        n_valid_trials=("trial_id", "count"),
        termination_overhead_frames=("termination_overhead_frames", "mean"),
        segmentation_purity_percent=("segmentation_purity_percent", "mean"),
        segment_length_std_dev=("n_frames", "std"),
        inference_accuracy=("inference_correct", "mean"),
    ).reset_index()

    summary["inference_accuracy"] = summary["inference_accuracy"] * 100.0
    summary = summary.rename(columns={"scenario": "Trigger Scenario"})
    return summary


def build_lighting_table(df: pd.DataFrame) -> pd.DataFrame:
    # ONLY explicitly allow lighting scenarios into this table
    target_scenarios = ["dim", "normal"]
    work = df[(df["valid_capture"] == True) & (df["scenario"].isin(target_scenarios))].copy()
    
    if work.empty:
        return pd.DataFrame()

    # Group by scenario directly
    grouped = work.groupby(["scenario", "method"], dropna=False)

    summary = grouped.agg(
        n_valid_trials=("trial_id", "count"),
        termination_overhead_frames=("termination_overhead_frames", "mean"),
        segmentation_purity_percent=("segmentation_purity_percent", "mean"),
        segment_length_std_dev=("n_frames", "std"),
        inference_accuracy=("inference_correct", "mean"),
    ).reset_index()

    summary["inference_accuracy"] = summary["inference_accuracy"] * 100.0
    summary = summary.rename(columns={"scenario": "Lighting Condition"})
    return summary


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ensure_dir(CONFIG["reports_dir"])

    trials = load_all_trials(CONFIG["data"]["baseline_dir"], CONFIG["data"]["proposed_dir"])
    if len(trials) == 0:
        print("No trials found. Run eval_merge.py first.")
        return

    baseline_trials = [t for t in trials if t["method"] == "baseline_manual" and t.get("gs") is not None]
    proposed_trials = [t for t in trials if t["method"] == "proposed_auto" and t.get("gs") is not None]

    print(f"Loaded trials: total={len(trials)}, baseline={len(baseline_trials)}, proposed={len(proposed_trials)}")

    if _PREDICT_FN is None:
        print("\n[INFO] Inference function not found.")
        print("[INFO] Segmentation metrics will still be computed.")
        print("[INFO] Inference accuracy will be NaN until you add:")
        print("       from gestureClassInference import predict_from_sequence")
        if _INFERENCE_IMPORT_ERROR:
            print(f"[INFO] Import error was: {_INFERENCE_IMPORT_ERROR}")

    # Optional plots
    m1 = run_metric1_plot(baseline_trials, proposed_trials, CONFIG["metric1"])
    if not m1["ok"]:
        print(m1["error"])

    m2 = run_metric2_plot(baseline_trials, proposed_trials, CONFIG["metric2"])
    if not m2["ok"]:
        print(m2["error"])

    # Trial-level metrics + inference
    df_trials = compute_trial_level_metrics(trials, CONFIG)
    ensure_dir(os.path.dirname(CONFIG["csv"]["trial_level_path"]))
    df_trials.to_csv(CONFIG["csv"]["trial_level_path"], index=False)

    # Final tables
    df_trigger = build_trigger_table(df_trials)
    df_lighting = build_lighting_table(df_trials)

    if not df_trigger.empty:
        df_trigger.to_csv(CONFIG["csv"]["trigger_table_path"], index=False)
        print("\n=== OBJECTIVE 2: TRIGGER-BASED RESULTS TABLE ===")
        print(df_trigger.to_string(index=False))
    else:
        print("\nNo valid trigger-based table could be generated.")

    if not df_lighting.empty:
        df_lighting.to_csv(CONFIG["csv"]["lighting_table_path"], index=False)
        print("\n=== OBJECTIVE 2: LIGHTING RESULTS TABLE ===")
        print(df_lighting.to_string(index=False))
    else:
        print("\nNo valid lighting table could be generated.")

    print("\nSaved reports:")
    print(f" - Trial-level CSV     : {CONFIG['csv']['trial_level_path']}")
    print(f" - Trigger table CSV   : {CONFIG['csv']['trigger_table_path']}")
    print(f" - Lighting table CSV  : {CONFIG['csv']['lighting_table_path']}")
    if m1.get("plot_path"):
        print(f" - Metric 1 plot       : {m1['plot_path']}")
    if m2.get("plot_path"):
        print(f" - Metric 2 plot       : {m2['plot_path']}")

    print("\nFinal paper tables expected:")
    print("Table 1: Trigger-Based Results")
    print("| Trigger Scenario | Method | Termination Overhead (frames) ↓ | Segmentation Purity (%) ↑ | Segment Length Std Dev ↓ | Inference Accuracy (%) ↑ |")
    print("\nTable 2: Lighting Results")
    print("| Lighting Condition | Method | Termination Overhead (frames) ↓ | Segmentation Purity (%) ↑ | Segment Length Std Dev ↓ | Inference Accuracy (%) ↑ |")


if __name__ == "__main__":
    main()