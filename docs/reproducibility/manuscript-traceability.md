# Manuscript-to-Code Traceability

This map follows the current C&EE manuscript. Table and figure numbers refer to that manuscript version. Paths are relative to this repository unless identified as external.

## Methods and System Figures

| Manuscript item | Retained implementation or evidence | Status | Boundary |
|---|---|---|---|
| Figure 1: HoloPi architecture | `main_orchestrator.py`; `raspi/camera/send_picamera_stream_to_server.py`; `mediamtx/`; `online_ASD/realtime/pipeline_main.py`; `Gesture_System/real-time-HGR-application/main.py`; `RAG/`; `STT/`; `Chatbot_Phi2/`; `TTS/`; `react_avatar/` | Mixed live-integrated and configuration-dependent | The figure shows LARA at its intended retrieval-stage placement, but the live RAG service does not invoke evaluated LARA. |
| Section 3: Edge I/O | `raspi/camera/send_picamera_stream_to_server.py`; `react_avatar/public/dome.html` | Live-integrated, hardware-dependent | Exact camera and microphone product models were not recorded. |
| Section 3: Media relay | `mediamtx/mediamtx.yml`; `mediamtx/mediamtx_watchdog.py`; WHIP/WHEP clients in the runtime modules | Live-integrated, configuration-dependent | MediaMTX is an external binary; network performance was not benchmarked. |
| Figure 2 and Section 4.1: LARA | `experiment_metric/reranker_metric/eval.ipynb` | Benchmark-only | The live `RAG/DatabaseRouting.py` uses ordinary batching and score sorting. |
| Figures 3--4 and Section 4.2: DART | `Gesture_System/real-time-HGR-application/hgr_box_gate_v2.py`; `main.py`; `gestureClassInference.py` | Live-integrated, classifier-dependent | The figures are explanatory diagrams, not generated directly by source code. |
| Figure 5 and Section 4.3: CATT-ASD | `online_ASD/utils/student_model.py`; `online_ASD/realtime/pipeline_main.py`; external MSSG `proposed/student_model.py` and `proposed/train_student.py` | Live-integrated runtime; external training workspace | The retained live repository contains inference code and weights, not the complete training/evaluation workspace. |
| Figure 6: CATT-ASD trade-off | Values from manuscript Table 5 | Descriptive manuscript visualization | It is not a separate experiment. Plot-generation source is maintained with the manuscript package. |

## Result Tables

| Table | Claim | Code/evidence location | Evidence status | Reproduction boundary |
|---|---|---|---|---|
| Table 1 | MS MARCO reranking effectiveness | `experiment_metric/reranker_metric/eval.ipynb` | Benchmark code and saved notebook output retained | Environment and remote dataset/model availability may affect a fresh rerun. |
| Table 2 | LARA padding and throughput measurements | Same LARA notebook; machine-readable reported values in manuscript Supplementary Data S2 | Reported single run | Exact original console output was not retained; a saved notebook run is closely matching, not identical. |
| Table 3 | DART spatial-trigger pilot | Manuscript Supplementary Data S1 | Reported-only | Complete three-participant trial data are unavailable. `metric_code/reports/` is a different, later single-session record. |
| Table 4 | DART lighting pilot | Manuscript Supplementary Data S1 | Reported-only | Same limitation as Table 3. The recovered report labels `normal`, whereas the manuscript's original protocol/results report `bright`; they must not be merged. |
| Table 5 | CATT-ASD architecture, metrics, and timing | External MSSG `proposed/comprehensive_eval.py` and `realtime/realtime_epoch/comprehensive_eval_result.json`; manuscript Supplementary Data S3 | CATT/MSSG machine-readable output retained externally and in supplement; LR-ASD reported-only | LR-ASD adapter, checkpoint, evaluator, and log are unavailable. Timing protocols are not fully identical across all rows. |
| Table 6 | Columbia speaker F1 | Same MSSG output and Supplementary Data S3 | CATT/MSSG retained; LR-ASD reported-only | Columbia contributed to CATT checkpoint selection and is cross-dataset validation, not an untouched external test. |
| Table 7 | CATT-ASD ablations | External MSSG comprehensive evaluator and output | Machine-readable output retained | Components were disabled at inference in one selected checkpoint; variants were not retrained. |

## External Revisions

- HoloPi integration repository baseline audited for the manuscript: `18de03cdf1c85251b8f9163b3e840ddaedb3826d`.
- MSSG retained main revision: `1db2924ccd973cb60321125b00f765c10611a322`.
- LARA notebook provenance revision cited by the manuscript supplement: `95a014e4c5435ae7c540f08ac7678e5daf145faa`.
- LR-ASD official release inspected during provenance review: `1b6dcd2d8fc2895683de6508ec6294ec47d388ca`.

The current cleanup is intentionally left uncommitted for the repository owner. After committing, record the new HoloPi commit alongside the baseline rather than replacing the historical baseline identifier.
