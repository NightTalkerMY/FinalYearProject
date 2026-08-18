# Dependency Records

HoloPi is a multi-environment repository. The retained requirement files came from different module environments and are not mutually compatible as one global environment.

| Scope | Requirement record | Interpretation |
|---|---|---|
| Orchestrator | `requirements.txt` | Retained root environment export |
| STT | `STT/requirements_STT.txt` | Small service-specific list |
| Phi-2 | `Chatbot_Phi2/requirements.txt` | Retained full environment export |
| RAG | `RAG/requirements.txt` | Retained full environment export |
| ZipVoice | `TTS/ZipVoice/requirements.txt` | ZipVoice environment/upstream dependencies |
| Legacy XTTS | `TTS/requirements_TTS.txt` | Older TTS path, not the current orchestrated ZipVoice service |
| DART | `Gesture_System/requirements.txt` | Retained full training/runtime environment export |
| CATT-ASD runtime | `online_ASD/requirements.txt` | Retained deployed inference environment export |
| LARA and metric tools | `experiment_metric/requirements.txt` | Curated experiment list with CUDA 12.1 PyTorch index |
| MediaMTX watchdog | `mediamtx/requirements.txt` | Requests-only environment export for the watchdog |

During the reproducibility cleanup, the seven UTF-16/CRLF requirement exports were converted mechanically to UTF-8/LF so that Git, editors, and package tooling can read them normally. Package names, versions, comments, and ordering were not changed.

## Installation Rule

Create a separate virtual environment for each runtime group. Do not concatenate these files. In particular, PyTorch, Transformers, NumPy, and related versions differ across DART, CATT-ASD, RAG, Phi-2, and the experiment workspace.

The retained files describe development environments, not independently validated minimal dependency sets. A successful `pip install` on a new machine still depends on:

- a compatible Python version;
- Windows and CUDA wheel availability;
- system libraries needed by audio/video packages;
- external model and dataset access;
- the hardware-specific deployment described in the root README.

## Historical Environment Boundaries

- LARA's manuscript run records Python 3.10.11, PyTorch 2.4.1 with CUDA 12.1, Sentence Transformers 5.2.2, and Transformers 5.0.0.
- The exact Python, PyTorch, and CUDA versions used for the reported CATT-ASD training run were not retained. The present `online_ASD/requirements.txt` describes the later retained inference environment and must not be used to infer missing historical versions.
- The complete DART manuscript raw trial data are unavailable, so dependency recovery cannot make Tables 3--4 independently recomputable.

Future cleanup may create minimal tested lockfiles, but those should be added only after fresh environment installation and module-level smoke tests on the target Windows/CUDA system.
