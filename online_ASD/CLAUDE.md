# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Online/real-time Active Speaker Detection (ASD) system. Uses a teacher-student architecture where 7 specialized CNN "teacher" encoders (face, face_large, face_small, face_body, face_body_large, face_down, background) extract multi-cue visual features, which are distilled into a lightweight `CausalStudentASD` model for real-time inference.

The system detects faces in video, tracks them across frames, extracts audio-visual features over a sliding window, and classifies each tracked speaker as speaking or silent.

## Architecture

### Teacher Models (training/feature extraction)
- **`ASD.py`** — Training harness for teacher ASD models. Two variants: `ASD` (with configurable encoder struct) and `ASD_face_audio` (face+audio with contrastive loss). Handles training loops, evaluation (mAP via sklearn), checkpoint loading (single or multi-model averaging of top-7 checkpoints).
- **`model/Model.py`** — Model definitions: `ASD_Model` (visual+audio encoders + BiGRU), `FrontASD_Model` (with graph-based classification via torch_geometric), `ASD_face_audio_Model` (face-only visual + audio).
- **`model/Encoder.py`** — Visual and audio CNN encoders using multi-scale convolution blocks (3x3 + 5x5 parallel paths). `visual_encoder` uses 3D convolutions (spatiotemporal), `audio_encoder` uses 2D convolutions on MFCC spectrograms. Encoder dimensions are parameterized via `encoder_struct` list (e.g., `[1, 32, 64, 128]`).
- **`model/Classifier.py`** — `BGRU`: Bidirectional GRU implemented as two unidirectional GRUs with manual flip, using GELU activation.
- **`model/Graph.py`** — `GraphASD`: Multi-stream spatial graph using torch_geometric (EdgeConv, SAGEConv) for inter-speaker reasoning. Has separate audio and visual message-passing paths.
- **`model/fusion.py`** — Attention-based feature fusion modules (AFF, iAFF, MS_CAM). Currently only used in `models_gnn.py`.
- **`loss.py`** — Loss functions: `lossAV` (audio-visual BCE), `lossV` (visual-only BCE), `lossContrast` (n-pair contrastive loss between audio and visual embeddings). Both lossAV and lossV use temperature scaling via parameter `r`.

### Student Model (real-time inference)
- **`utils/student_model.py`** — `CausalStudentASD`: Lightweight model for Raspberry Pi deployment. Takes concatenated 896-dim visual features (7 cues x 128) + 128-dim audio. Pipeline: linear projection -> GRU (unidirectional, causal) -> multi-head self-attention with causal mask -> `CausalGraphNet` (pure PyTorch, no torch_geometric) -> classifier. Input shape: `[B, 5 speakers, 50 frames, features]` with a validity mask for ghost speakers.

### Real-Time Pipeline
- **`realtime/realtime_main.py`** — End-to-end pipeline. Supports two modes via `MODE` variable: `"REMOTE"` (Tailscale TCP streaming) and `"FILE"` (local video/audio files). Uses S3FD face detection, IoU-based tracking, 50-frame sliding window, background inference thread, and EMA-smoothed prediction display.
- **`utils/data_streamer.py`** — Video/audio I/O. `RemoteVideoClient`/`RemoteAudioClient` stream over TCP sockets (pickle-serialized frames). `FileVideoClient`/`FileAudioClient` read local files. Audio client maintains a thread-safe rolling buffer (10s at 16kHz).
- **`utils/online_tracker.py`** — Simple IoU-based multi-object tracker with EMA bbox smoothing (alpha=0.8) and configurable max-lost-frames tolerance.
- **`model/faceDetector/`** — S3FD face detector with pretrained weights (`sfd_face.pth`).

### Pretrained Weights
- `pretrain_model/` — Contains 7 teacher model checkpoint directories (each with ~7 epoch snapshots for model averaging) plus `holopi_student_best.pt` (distilled student weights).

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run real-time pipeline (edit MODE/VIDEO_FILE/AUDIO_FILE in realtime_main.py first)
python realtime/realtime_main.py
```

## Key Constants

- `WINDOW_SIZE = 50` — Sliding window length (frames) for sequence models
- `MAX_SPEAKERS = 5` — Maximum simultaneous tracked speakers
- `AUDIO_SAMPLE_RATE = 16000` — Expected audio sample rate
- `encoder_struct = [32, 64, 128]` — Default teacher encoder channel dimensions (prepended with 1 for input channels)
- Visual normalization: `(x / 255 - 0.4161) / 0.1688`
- Audio: 13-dim MFCC features, 25ms window, 10ms step, padded/truncated to 200 frames

## Dependencies

- PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121
- torch_geometric 2.7.0 (used by `model/Graph.py` and `model/models_gnn.py`; NOT used by student model)
- OpenCV, python_speech_features, scikit-learn, scipy, pandas
- S3FD face detector (bundled with pretrained weights)

## Crop Geometry

The 7 visual cues use specific crop offsets relative to the detected face bounding box (with 1.2x padding). These offsets match the offline data preparation scripts (`prepare_*.py`, not in this repo). See `realtime_main.py:276-293` for the exact formulas. Changing these breaks compatibility with pretrained weights.
