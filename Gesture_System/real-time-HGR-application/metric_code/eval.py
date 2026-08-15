# eval.py
# ------------------------------------------------------------
# OBJECTIVE 2 - FINAL EVALUATION (SEGMENTATION + INFERENCE)
#
# Pipeline:
#   eval_merge.py -> saves gs_sequence.npy + meta.json
#   eval.py       -> runs segmentation metrics + gesture inference
#
# Metrics:
#   1. Activation Reliability (%)         ↑  (auto-methods only; baseline = N/A)
#   2. Segmentation Purity (%)            ↑  (relative to own peak; secondary context)
#   3. Recognition Efficiency (%)         ↑  (correct + compact = high; missed/wrong = 0)
#   4. Inference Accuracy (%)             ↑  (PRIMARY evaluation metric)
#
# Final outputs:
#   - Trigger-based table CSV
#   - Lighting-based table CSV
#   - Trial-level CSV
#   - Optional plots
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONFIG = {
    "data": {
        "baseline_dir": "./metric_tests/baseline_manual",
        "proposed_dir": "./metric_tests/proposed_auto",
        "reference_dir": "./metric_tests/reference_spotter",
    },
    "efficiency": {
        "norm_window": 30,  # Reference method's fixed window — normalisation anchor
        "save_plot": True,
        "plot_path": "./reports/objective2_metric3_recognition_efficiency.png",
    },
    "purity": {
        "action_move_thresh": 0.15,
        "save_plot": True,
        "plot_path": "./reports/objective2_metric2_purity_boxplot.png",
    },
    "inference": {
        "enabled": True,
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
    if "/reference_spotter/" in p:
        return "reference_spotter"
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

    root_map = {"baseline_manual": "baseline_manual", "proposed_auto": "proposed_auto", "reference_spotter": "reference_spotter"}
    method_root_name = root_map.get(method, "proposed_auto")
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


def load_all_trials(baseline_root: str, proposed_root: str, reference_root: str | None = None) -> list[dict]:
    trials = []
    for trial_dir in discover_trial_dirs(baseline_root):
        t = load_trial(trial_dir)
        if t is not None:
            trials.append(t)
    for trial_dir in discover_trial_dirs(proposed_root):
        t = load_trial(trial_dir)
        if t is not None:
            trials.append(t)
    if reference_root:
        for trial_dir in discover_trial_dirs(reference_root):
            t = load_trial(trial_dir)
            if t is not None:
                trials.append(t)
    return trials


# ------------------------------------------------------------
# SEQUENCE PROCESSING
# ------------------------------------------------------------
SCALE_WEIGHT = 5.0  # Scale changes are ~5x smaller than wrist displacement


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


def motion_signal_from_trial(trial: dict, wrist_idx: int = 0) -> np.ndarray | None:
    """
    Combined per-frame motion signal: max(wrist_vel, scale_vel * SCALE_WEIGHT).

    This correctly captures BOTH gesture types:
      - Swipes: wrist velocity dominates
      - Grab/Expand: scale velocity (fingertip-to-wrist distance change) dominates

    Returns array of length (n_frames - 1), or None.
    """
    gs = trial.get("gs")
    if gs is None or not isinstance(gs, np.ndarray) or gs.ndim != 3:
        return None
    if len(gs) < 2 or gs.shape[1] < 21 or gs.shape[2] < 3:
        return None

    # Wrist velocity (inter-frame displacement)
    wrist = gs[:, wrist_idx, :3]
    wrist_vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)

    # Scale velocity (change in mean fingertip-to-wrist distance)
    tips = gs[:, [4, 8, 12, 16, 20], :]
    scales = np.mean(
        np.linalg.norm(tips - wrist[:, np.newaxis, :], axis=2), axis=1
    )
    scale_vel = np.abs(np.diff(scales))

    # Combine: take max of both channels (scale weighted up to match wrist magnitude)
    combined = np.maximum(wrist_vel, scale_vel * SCALE_WEIGHT)
    return combined if combined.size else None


def velocity_sequence_from_trial(trial: dict, wrist_idx: int = 0) -> np.ndarray | None:
    """Use combined motion signal instead of wrist-only."""
    return motion_signal_from_trial(trial, wrist_idx=wrist_idx)


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




# ------------------------------------------------------------
# OPTIONAL PLOTS
# ------------------------------------------------------------
def _trial_recognition_efficiency(trial: dict, norm_window: float) -> float | None:
    """Per-trial Recognition Efficiency for plot data (valid trials only)."""
    n = segment_length_frames(trial)
    if n <= 0:
        return None
    return min(norm_window / n, 1.0) * 100.0


def run_efficiency_plot(
    baseline_trials: list[dict],
    proposed_trials: list[dict],
    cfg: dict,
    reference_trials: list[dict] | None = None,
) -> dict:
    """Boxplot of Recognition Efficiency (%) per method."""
    norm_w = float(cfg.get("norm_window", 30))

    base_vals = [v for t in baseline_trials if (v := _trial_recognition_efficiency(t, norm_w)) is not None]
    prop_vals = [v for t in proposed_trials if (v := _trial_recognition_efficiency(t, norm_w)) is not None]

    has_ref = reference_trials and len(reference_trials) > 0
    ref_vals = []
    if has_ref:
        ref_vals = [v for t in reference_trials if (v := _trial_recognition_efficiency(t, norm_w)) is not None]
        if not ref_vals:
            has_ref = False

    if len(base_vals) == 0 or len(prop_vals) == 0:
        return {"ok": False, "error": "Efficiency plot: no valid values found."}

    base_mean, base_std = float(np.mean(base_vals)), float(np.std(base_vals))
    prop_mean, prop_std = float(np.mean(prop_vals)), float(np.std(prop_vals))

    box_data = [base_vals]
    box_labels = ["Baseline"]
    if has_ref:
        box_data.append(ref_vals)
        box_labels.append("Reference")
    box_data.append(prop_vals)
    box_labels.append("Proposed")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)

    rng = np.random.default_rng(42)
    colors = ["tab:blue", "tab:green", "tab:orange"] if has_ref else ["tab:blue", "tab:orange"]
    all_vals = [base_vals] + ([ref_vals] if has_ref else []) + [prop_vals]

    for i, (vals, color) in enumerate(zip(all_vals, colors), start=1):
        ax.scatter(i + rng.normal(0, 0.03, size=len(vals)), vals, alpha=0.5, s=18, color=color)

    ax.set_title("Recognition Efficiency", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Efficiency (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    txt_lines = [
        f"Norm window: {norm_w:.0f} frames",
        f"Baseline:  {base_mean:.1f}% +/- {base_std:.1f}% (n={len(base_vals)})",
    ]
    if has_ref:
        ref_m, ref_s = float(np.mean(ref_vals)), float(np.std(ref_vals))
        txt_lines.append(f"Reference: {ref_m:.1f}% +/- {ref_s:.1f}% (n={len(ref_vals)})")
    txt_lines.append(f"Proposed:  {prop_mean:.1f}% +/- {prop_std:.1f}% (n={len(prop_vals)})")
    txt = "\n".join(txt_lines)

    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()
    if cfg.get("save_plot", False):
        ensure_dir(os.path.dirname(cfg["plot_path"]))
        plt.savefig(cfg["plot_path"], dpi=200)
    plt.show()

    result = {
        "ok": True,
        "baseline_n_valid": len(base_vals),
        "proposed_n_valid": len(prop_vals),
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }
    if has_ref:
        result["reference_n_valid"] = len(ref_vals)
    return result


def run_metric2_plot(
    baseline_trials: list[dict],
    proposed_trials: list[dict],
    cfg: dict,
    reference_trials: list[dict] | None = None,
) -> dict:
    thresh = float(cfg["action_move_thresh"])

    base_vals = [v for t in baseline_trials if (v := purity_percent_from_trial(t, thresh)) is not None]
    prop_vals = [v for t in proposed_trials if (v := purity_percent_from_trial(t, thresh)) is not None]

    has_ref = reference_trials and len(reference_trials) > 0
    ref_vals = []
    if has_ref:
        ref_vals = [v for t in reference_trials if (v := purity_percent_from_trial(t, thresh)) is not None]
        if not ref_vals:
            has_ref = False

    if len(base_vals) == 0 or len(prop_vals) == 0:
        return {"ok": False, "error": "Metric 2 plot: no valid purity values found."}

    base_mean, base_std = float(np.mean(base_vals)), float(np.std(base_vals))
    prop_mean, prop_std = float(np.mean(prop_vals)), float(np.std(prop_vals))

    # Build box data and labels
    box_data = [base_vals]
    box_labels = ["Baseline"]
    if has_ref:
        box_data.append(ref_vals)
        box_labels.append("Reference")
    box_data.append(prop_vals)
    box_labels.append("Proposed")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)

    rng = np.random.default_rng(42)
    colors = ["tab:blue", "tab:green", "tab:orange"] if has_ref else ["tab:blue", "tab:orange"]
    all_vals = [base_vals] + ([ref_vals] if has_ref else []) + [prop_vals]

    for i, (vals, color) in enumerate(zip(all_vals, colors), start=1):
        ax.scatter(i + rng.normal(0, 0.03, size=len(vals)), vals, alpha=0.5, s=18, color=color)

    ax.set_title("Metric 2: Segmentation Purity", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Purity (%) = ActionSteps / TotalSteps × 100", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    txt_lines = [
        "Segmentation purity summary",
        f"Action threshold: {thresh:g}",
        f"Baseline:  {base_mean:.2f}% +/- {base_std:.2f}% (n={len(base_vals)})",
    ]
    if has_ref:
        ref_mean, ref_std = float(np.mean(ref_vals)), float(np.std(ref_vals))
        txt_lines.append(f"Reference: {ref_mean:.2f}% +/- {ref_std:.2f}% (n={len(ref_vals)})")
    txt_lines.append(f"Proposed:  {prop_mean:.2f}% +/- {prop_std:.2f}% (n={len(prop_vals)})")
    txt = "\n".join(txt_lines)

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

    result = {
        "ok": True,
        "baseline_n_valid": len(base_vals),
        "proposed_n_valid": len(prop_vals),
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }
    if has_ref:
        result["reference_n_valid"] = len(ref_vals)
    return result


# ------------------------------------------------------------
# TRIAL-LEVEL METRICS
# ------------------------------------------------------------
def compute_trial_level_metrics(trials: list[dict], cfg: dict) -> pd.DataFrame:
    purity_thresh = float(cfg["purity"]["action_move_thresh"])
    norm_window = float(cfg["efficiency"]["norm_window"])

    rows = []
    for t in trials:
        meta = t.get("meta", {})
        gs = t.get("gs")

        is_valid = meta.get("valid_capture", True)
        is_missed = meta.get("missed", False)
        n_frames = segment_length_frames(t)

        gesture_label = meta.get(CONFIG["inference"]["truth_key"])

        if is_valid and not is_missed:
            # Normal trial — compute real metrics
            purity = purity_percent_from_trial(t, purity_thresh)

            predicted_label = None
            if gs is not None:
                predicted_label = infer_label_from_sequence(gs)

            inference_correct = None
            if gesture_label is not None and predicted_label is not None:
                inference_correct = int(str(gesture_label) == str(predicted_label))

            # Recognition Efficiency: correct + compact = high score
            if inference_correct == 1 and n_frames > 0:
                recognition_efficiency = min(norm_window / n_frames, 1.0) * 100.0
            else:
                recognition_efficiency = 0.0
        else:
            # Missed trial — penalize with 0 so the denominator stays fair
            # (prevents survivorship bias when averaging across methods)
            purity = 0.0
            predicted_label = None
            inference_correct = 0
            recognition_efficiency = 0.0

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
            "valid_capture": is_valid,
            "missed": is_missed,
            "n_frames": n_frames,
            "segmentation_purity_percent": purity,
            "predicted_label": predicted_label,
            "inference_correct": inference_correct,
            "recognition_efficiency": recognition_efficiency,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# ACTIVATION RELIABILITY
# ------------------------------------------------------------
def compute_activation_reliability(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Activation Reliability = successful triggers / total attempts per method.

    Total attempts per (group, gesture) = number of baseline trials for that group
    (baseline always fires because it's manual SPACE).
    Missed trials have valid_capture=False.
    """
    target = df[group_col].dropna().unique().tolist()
    all_trials = df[df[group_col].isin(target)].copy()

    if all_trials.empty:
        return pd.DataFrame()

    # Total attempts per (group, gesture) = baseline count
    base = all_trials[all_trials["method"] == "baseline_manual"]
    attempt_counts = base.groupby([group_col, "gesture_label"])["trial_id"].count()
    attempt_counts.name = "total_attempts"

    rows = []
    for method in all_trials["method"].unique():
        method_df = all_trials[all_trials["method"] == method]
        for grp in target:
            grp_df = method_df[method_df[group_col] == grp]
            total = 0
            successful = 0
            for gesture in base[base[group_col] == grp]["gesture_label"].unique():
                n_attempts = int(attempt_counts.get((grp, gesture), 0))
                n_success = int(grp_df[(grp_df["gesture_label"] == gesture) & (grp_df["valid_capture"] == True)].shape[0])
                total += n_attempts
                successful += n_success
            reliability = (successful / total * 100.0) if total > 0 else 0.0
            rows.append({
                group_col: grp,
                "method": method,
                "activation_reliability_percent": round(reliability, 1),
                "successful_triggers": successful,
                "total_attempts": total,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# FINAL TABLES
# ------------------------------------------------------------
def _build_summary_table(df: pd.DataFrame, target_scenarios: list[str], group_col: str) -> pd.DataFrame:
    """Shared logic for trigger-based and lighting-based summary tables."""
    all_trials = df[df["scenario"].isin(target_scenarios)].copy()
    if all_trials.empty:
        return pd.DataFrame()

    # ---- Purity, Efficiency & Inference: use ALL trials (missed = 0, penalizes low reliability) ----
    grouped_all = all_trials.groupby(["scenario", "method"], dropna=False)

    summary = grouped_all.agg(
        n_total_trials=("trial_id", "count"),
        segmentation_purity_percent=("segmentation_purity_percent", "mean"),
        recognition_efficiency=("recognition_efficiency", "mean"),
        inference_accuracy=("inference_correct", "mean"),
    ).reset_index()

    summary["inference_accuracy"] = summary["inference_accuracy"] * 100.0

    # Merge activation reliability (auto-methods only; baseline = N/A)
    reliability_df = compute_activation_reliability(all_trials, "scenario")
    if not reliability_df.empty:
        summary = summary.merge(
            reliability_df[["scenario", "method", "activation_reliability_percent"]],
            on=["scenario", "method"],
            how="left",
        )
    else:
        summary["activation_reliability_percent"] = np.nan

    # Baseline is always 100% by definition (manual SPACE) — mark as N/A
    summary.loc[
        summary["method"] == "baseline_manual", "activation_reliability_percent"
    ] = np.nan

    # Reorder columns for readability
    col_order = [
        "scenario", "method", "n_total_trials",
        "activation_reliability_percent",
        "segmentation_purity_percent",
        "recognition_efficiency",
        "inference_accuracy",
    ]
    summary = summary[[c for c in col_order if c in summary.columns]]
    summary = summary.rename(columns={"scenario": group_col})
    return summary


def build_trigger_table(df: pd.DataFrame) -> pd.DataFrame:
    return _build_summary_table(df, ["wrist_trigger", "scale_trigger"], "Trigger Scenario")


def build_lighting_table(df: pd.DataFrame) -> pd.DataFrame:
    return _build_summary_table(df, ["dim", "normal"], "Lighting Condition")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ensure_dir(CONFIG["reports_dir"])

    trials = load_all_trials(
        CONFIG["data"]["baseline_dir"],
        CONFIG["data"]["proposed_dir"],
        CONFIG["data"].get("reference_dir"),
    )
    if len(trials) == 0:
        print("No trials found. Run eval_merge.py first.")
        return

    # Separate valid trials (with gs data) for plots
    baseline_trials = [t for t in trials if t["method"] == "baseline_manual" and t.get("gs") is not None]
    proposed_trials = [t for t in trials if t["method"] == "proposed_auto" and t.get("gs") is not None]
    reference_trials = [t for t in trials if t["method"] == "reference_spotter" and t.get("gs") is not None]

    # Count missed trials
    missed_prop = sum(1 for t in trials if t["method"] == "proposed_auto" and t.get("meta", {}).get("missed"))
    missed_ref = sum(1 for t in trials if t["method"] == "reference_spotter" and t.get("meta", {}).get("missed"))

    print(f"Loaded trials: total={len(trials)}, baseline={len(baseline_trials)}, "
          f"reference={len(reference_trials)} (+{missed_ref} missed), "
          f"proposed={len(proposed_trials)} (+{missed_prop} missed)")

    if _PREDICT_FN is None:
        print("\n[INFO] Inference function not found.")
        print("[INFO] Segmentation metrics will still be computed.")
        print("[INFO] Inference accuracy will be NaN until you add:")
        print("       from gestureClassInference import predict_from_sequence")
        if _INFERENCE_IMPORT_ERROR:
            print(f"[INFO] Import error was: {_INFERENCE_IMPORT_ERROR}")

    ref_or_none = reference_trials if reference_trials else None

    # Plot 1: Recognition Efficiency
    m1 = run_efficiency_plot(baseline_trials, proposed_trials, CONFIG["efficiency"], ref_or_none)
    if not m1["ok"]:
        print(m1["error"])

    # Plot 2: Segmentation Purity
    m2 = run_metric2_plot(baseline_trials, proposed_trials, CONFIG["purity"], ref_or_none)
    if not m2["ok"]:
        print(m2["error"])

    # Trial-level metrics + inference (includes missed trials for reliability)
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
        print(f" - Efficiency plot     : {m1['plot_path']}")
    if m2.get("plot_path"):
        print(f" - Purity plot         : {m2['plot_path']}")

    print("\nFinal paper table columns (3 methods: baseline_manual, reference_spotter, proposed_auto):")
    print("| Scenario | Method | Activation Reliability % ↑ | Seg. Purity % ↑ | Recognition Efficiency % ↑ | Inference Accuracy % ↑ |")


if __name__ == "__main__":
    main()
