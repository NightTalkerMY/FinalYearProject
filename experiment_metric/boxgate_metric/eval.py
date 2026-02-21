# eval.py
import os
import glob
import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch


CONFIG = {
    "data": {
        "baseline_dir": "./metric_tests/baseline_manual",
        "proposed_dir": "./metric_tests/proposed_auto",
        "recursive": True,
    },
    "metric1": {
        "stillness_thresh": 0.01,
        "inset_last_n": 20,
        "inset_ylim": (0.0, 0.05),
        "main_ylim": (0.0, 0.30),
        "save_plot": True,
        "plot_path": "./reports/metric1_velocity_inset.png",
        "max_left_steps": None,
    },
    "metric2": {
        "action_move_thresh": 0.001,
        "save_plot": True,
        "plot_path": "./reports/metric2_purity_boxplot.png",
    },
    "reports_dir": "./reports",
}


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_npy_files(folder: str, recursive: bool = True) -> list[str]:
    pattern = os.path.join(folder, "**", "*.npy") if recursive else os.path.join(folder, "*.npy")
    return glob.glob(pattern, recursive=recursive)


def load_wrist_positions(npy_path: str, wrist_idx: int = 0) -> np.ndarray | None:
    try:
        data = np.load(npy_path)
    except Exception:
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


def _maybe_pick_gs_sequence(paths: list[str]) -> list[str]:
    gs = [p for p in paths if os.path.basename(p) == "gs_sequence.npy"]
    return gs if gs else paths


def velocity_sequence_from_file(npy_path: str, wrist_idx: int = 0) -> np.ndarray | None:
    wrist = load_wrist_positions(npy_path, wrist_idx=wrist_idx)
    if wrist is None or len(wrist) < 5:
        return None

    v = step_displacement(wrist)
    return v if v.size else None


def build_end_aligned_matrix(files: list[str]) -> tuple[np.ndarray, int] | tuple[None, None]:
    seqs = []
    for f in files:
        v = velocity_sequence_from_file(f, wrist_idx=0)
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
        std  = np.nanstd(mat, axis=0)

    n = np.sum(~np.isnan(mat), axis=0)
    return mean, std, n




def run_metric1(baseline_files: list[str], proposed_files: list[str], cfg: dict) -> dict:
    still_thresh   = float(cfg["stillness_thresh"])
    inset_last_n   = int(cfg["inset_last_n"])
    inset_ylim     = tuple(cfg["inset_ylim"])
    main_ylim      = tuple(cfg["main_ylim"])
    max_left_steps = cfg.get("max_left_steps", None)

    baseline_files = _maybe_pick_gs_sequence(baseline_files)
    proposed_files = _maybe_pick_gs_sequence(proposed_files)

    # --- NEW: wasted-frames (tail below threshold) stats ---
    def wasted_tail_frames(files: list[str], thresh: float) -> tuple[int, float, int]:
        wasted_list = []
        for f in files:
            v = velocity_sequence_from_file(f, wrist_idx=0)
            if v is None or v.size == 0:
                continue

            # Find last index where motion is ABOVE threshold
            idx = np.where(v > thresh)[0]
            if idx.size == 0:
                wasted = int(v.size)  # never moved above thresh
            else:
                last_active = int(idx[-1])
                wasted = int(v.size - 1 - last_active)  # tail length after last active step

            wasted_list.append(wasted)

        total = int(np.sum(wasted_list)) if wasted_list else 0
        mean  = float(np.mean(wasted_list)) if wasted_list else 0.0
        n     = int(len(wasted_list))
        return total, mean, n

    base_wasted_total, base_wasted_mean, base_wasted_n = wasted_tail_frames(baseline_files, still_thresh)
    prop_wasted_total, prop_wasted_mean, prop_wasted_n = wasted_tail_frames(proposed_files, still_thresh)

    base_mat, _ = build_end_aligned_matrix(baseline_files)
    prop_mat, _ = build_end_aligned_matrix(proposed_files)

    if base_mat is None or prop_mat is None:
        return {"ok": False, "error": "Metric 1: no valid .npy files found (or wrong shape)."}

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

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x, base_mean, linewidth=2, label="Baseline (Manual Stop)")
    ax.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.15)

    ax.plot(x, prop_mean, linewidth=2, label="Proposed (BoxGate Stop)")
    ax.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.15)

    # --- FIX: bring back threshold labelling (legend entry) ---
    ax.axhline(
        y=still_thresh,
        linestyle=":",
        linewidth=1.5,
        color="red",
        label=f"Stillness Threshold ({still_thresh:g})"
    )

    ax.set_title("Metric 1: Termination Overhead", fontsize=14, pad=16, fontweight="bold")
    ax.set_ylabel("Wrist Motion Magnitude (Inter-frame Displacement)", fontsize=12)
    ax.set_xlabel("Number of frames overtime", fontsize=12)
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



    # --- Place text box directly under the legend, perfectly aligned ---
    fig.canvas.draw()  # needed to get correct legend bbox
    renderer = fig.canvas.get_renderer()
    bbox_disp = leg.get_window_extent(renderer=renderer)

    # use legend RIGHT edge instead of left
    (x1, y0) = ax.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y0))

    x_offset = 0.00
    y_offset = 0.01

    ax.text(
        x1 - x_offset, y0 - y_offset, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top", ha="right",   # <-- THIS is the key
        zorder=20,
        bbox=dict(
            facecolor="white",
            alpha=0.92,
            edgecolor="0.75",
            boxstyle="round,pad=0.4"
        )
    )






    # -------------------------------
    # INSET (restore highlight + keep visible)
    # -------------------------------
    axins = ax.inset_axes([0.24, 0.57, 0.52, 0.26])

    axins.plot(x, base_mean, linewidth=2)
    axins.fill_between(x, base_mean - base_std, base_mean + base_std, alpha=0.20)

    axins.plot(x, prop_mean, linewidth=2)
    axins.fill_between(x, prop_mean - prop_std, prop_mean + prop_std, alpha=0.20)

    axins.axhline(y=still_thresh, linestyle=":", linewidth=1.5, color="red")

    axins.set_xlim(-inset_last_n, 0)
    axins.set_ylim(inset_ylim[0], inset_ylim[1])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(f"Last {inset_last_n} Frames", fontsize=10)

    # --- FIX: highlight was getting visually “lost” due to inset patch settings ---
    # Put highlight first and keep inset background slightly transparent so it stays visible.
    axins.axhspan(
        inset_ylim[0],
        still_thresh,
        color="red",
        alpha=0.18,
        zorder=0
    )
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(0.90)
    axins.set_zorder(10)

    pp, p1, p2 = mark_inset(
        ax, axins,
        loc1=2, loc2=4,
        fc="none", ec="0.5", lw=1
    )
    p1.set_visible(False)
    p2.set_visible(False)

    xyA = (1, 0)
    xB = -inset_last_n / 2
    yB = inset_ylim[1]
    xyB = (xB, yB)

    con = ConnectionPatch(
        xyA=xyA, coordsA=axins.transAxes,
        xyB=xyB, coordsB=ax.transData,
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
        "baseline_trials": int(base_mat.shape[0]),
        "proposed_trials": int(prop_mat.shape[0]),
        "max_len_steps": int(max_len),
        "baseline_wasted_total": base_wasted_total,
        "proposed_wasted_total": prop_wasted_total,
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }


def purity_percent(npy_path: str, action_move_thresh: float) -> float | None:
    wrist = load_wrist_positions(npy_path, wrist_idx=0)
    if wrist is None or len(wrist) < 2:
        return None
    steps = step_displacement(wrist)
    if len(steps) == 0:
        return None
    return float((np.sum(steps > action_move_thresh) / len(steps)) * 100.0)


def run_metric2(baseline_files: list[str], proposed_files: list[str], cfg: dict) -> dict:
    thresh = float(cfg["action_move_thresh"])

    baseline_files = _maybe_pick_gs_sequence(baseline_files)
    proposed_files = _maybe_pick_gs_sequence(proposed_files)

    base_vals = []
    for f in baseline_files:
        v = purity_percent(f, thresh)
        if v is not None:
            base_vals.append(v)

    prop_vals = []
    for f in proposed_files:
        v = purity_percent(f, thresh)
        if v is not None:
            prop_vals.append(v)

    if len(base_vals) == 0 or len(prop_vals) == 0:
        return {"ok": False, "error": "Metric 2: no valid purity values (check units/shape)."}

    base_mean, base_std = float(np.mean(base_vals)), float(np.std(base_vals))
    prop_mean, prop_std = float(np.mean(prop_vals)), float(np.std(prop_vals))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot([base_vals, prop_vals], tick_labels=["Baseline", "Proposed"], showmeans=True)

    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.normal(0, 0.03, size=len(base_vals)), base_vals, alpha=0.5, s=18)
    ax.scatter(2 + rng.normal(0, 0.03, size=len(prop_vals)), prop_vals, alpha=0.5, s=18)

    ax.set_title("Metric 2: Segmentation Purity (Content-to-Container Ratio)", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Purity (%) = ActionSteps / TotalSteps × 100", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    txt = (
        "Segmentation purity summary\n"
        f"Action threshold: {thresh:g}\n"
        f"Baseline: {base_mean:.2f}% ± {base_std:.2f}% (n={len(base_vals)})\n"
        f"Proposed: {prop_mean:.2f}% ± {prop_std:.2f}% (n={len(prop_vals)})"
    )


    # Top-right corner (consistent right-aligned style)
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top", ha="right",
        bbox=dict(
            facecolor="white",
            alpha=0.92,
            edgecolor="0.75",
            boxstyle="round,pad=0.4"
        )
    )

    plt.tight_layout()
    if cfg.get("save_plot", False):
        ensure_dir(os.path.dirname(cfg["plot_path"]))
        plt.savefig(cfg["plot_path"], dpi=200)
    plt.show()

    return {
        "ok": True,
        "action_move_thresh": thresh,
        "baseline_n_valid": len(base_vals),
        "proposed_n_valid": len(prop_vals),
        "baseline_mean_purity": base_mean,
        "proposed_mean_purity": prop_mean,
        "plot_path": cfg.get("plot_path") if cfg.get("save_plot") else None,
    }



def main():
    ensure_dir(CONFIG["reports_dir"])

    baseline_files = list_npy_files(CONFIG["data"]["baseline_dir"], CONFIG["data"]["recursive"])
    proposed_files = list_npy_files(CONFIG["data"]["proposed_dir"], CONFIG["data"]["recursive"])

    run_metric1(baseline_files, proposed_files, CONFIG["metric1"])
    run_metric2(baseline_files, proposed_files, CONFIG["metric2"])


if __name__ == "__main__":
    main()
