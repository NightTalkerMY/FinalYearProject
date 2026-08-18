# Experiment Workspaces

This directory contains module-level experiments. Its contents are not automatically invoked by `main_orchestrator.py`.

## LARA Reranking

`reranker_metric/eval.ipynb` is the retained implementation of the manuscript's evaluated LARA procedure, including length-aware packing before cross-encoder inference and reciprocal rank anchoring after scoring.

The notebook supports the effectiveness evaluation and contains a closely matching saved hardware run. The exact console output behind the single-run manuscript Table 2 transcription was not retained, so a new timing run is expected to differ slightly.

`reranker_metric/retail_qna_eval_100.json` has no relevance labels. It supports efficiency evaluation only.

## Boxgate/DART Copies

`boxgate_metric/` is an earlier retained evaluation-code copy. The currently documented DART runtime and legacy report workspace are under `../Gesture_System/real-time-HGR-application/metric_code/`.

Neither location contains the complete three-participant raw trials behind manuscript Tables 3--4. Do not combine the recovered single-session reports with the manuscript aggregates.

## Environment

Use a separate environment based on `requirements.txt`. The LARA manuscript run records Windows, an NVIDIA RTX 4080 SUPER, Python 3.10.11, PyTorch 2.4.1 with CUDA 12.1, Sentence Transformers 5.2.2, and Transformers 5.0.0.

See `../docs/reproducibility/` before interpreting or publishing new results.
