# DART Gesture System

This module contains training code, the deployed real-time hand-gesture runtime, and evaluation utilities. These scopes should be treated separately.

## Live Runtime

The orchestrator launches `real-time-HGR-application/main.py`. It consumes the MediaMTX `cam1` stream by WHEP, applies the DART gate implemented in `hgr_box_gate_v2.py`, and passes retained sequences to `gestureClassInference.py`. Recognized gestures are posted to the orchestrator's `/gesture_command` endpoint.

The classifier export under `real-time-HGR-application/.sources/` is ignored by Git and must be supplied separately. Runtime endpoints can be overridden using the variables documented in the repository-level `.env.example`.

## Training Workspace

`data_training/` contains the retained gesture-classifier development code. Its datasets and checkpoints are external or ignored and are not a fresh-clone training reproduction.

## Evaluation Workspace

`real-time-HGR-application/metric_code/` contains capture and metric programs. `eval.py` expects the associated trial directories and produces trial-level and aggregate reports.

The checked-in `metric_code/reports/` files belong to a genuine, independently recovered legacy single-session run from May 2026. They do not match the three-participant experiment reported in manuscript Tables 3--4 and must not be used to recompute those values. The manuscript aggregates survive as reported-only Supplementary Data S1; the complete original trial files are unavailable.

See `../docs/reproducibility/runtime-and-experiments.md` for the complete boundary.
