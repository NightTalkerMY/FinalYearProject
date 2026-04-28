# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intelligent Holographic AI retail assistant ("PUMA Holographic Orchestrator"). A distributed microservice system that integrates an LLM chatbot, retrieval-augmented generation, gesture control, speech processing, active speaker detection, and a 3D React avatar — all coordinated by a central FastAPI orchestrator.

## Architecture

### Central Orchestrator (`main_orchestrator.py`)
- FastAPI server on port 5000 that manages the entire system lifecycle
- Launches and monitors all subservices: MediaMTX watchdog, React frontend, and AI microservices
- AI services are launched only after both the avatar stream and Pi camera stream are verified stable (4s stability check)
- Exposes the voice/text processing pipeline: STT → RAG → LLM → TTS, plus endpoints for gesture relay, carousel focus tracking, and state polling
- Global `SYSTEM_STATE` dict tracks audio URLs, viseme data, gesture commands, carousel/product state (ASINs), and stream health

### AI Microservices (each has its own venv and `main.py`)
| Service | Directory | Port | Tech |
|---------|-----------|------|------|
| STT | `STT/` | 8000 | OpenAI Whisper |
| LLM | `Chatbot_Phi2/` | 8001 | Microsoft Phi-2 |
| RAG | `RAG/` | 8002 | ChromaDB + instruction-tuned semantic routing |
| TTS | `TTS/ZipVoice/` | 8003 | ZipVoice (formerly Coqui XTTS) |
| Lip Sync | `TTS/allosaurus/` | 8004 | Allosaurus viseme server (`server.py`) |
| Gesture | `Gesture_System/real-time-HGR-application/` | 8889 | ResNet hand gesture recognition |

### Online Active Speaker Detection (`online_ASD/`)
Separate subsystem with its own CLAUDE.md. Teacher-student CNN architecture (7 teacher encoders distilled into `CausalStudentASD`) for real-time face detection and speaker classification. See `online_ASD/CLAUDE.md` for full details.

### Frontend (`react_avatar/`)
React + Vite app rendering a 3D avatar. `launch-hologram.js` manages the Puppeteer-based hologram display. Streams via MediaMTX + FFmpeg.

### Supporting Infrastructure
- `mediamtx/` — MediaMTX config + `mediamtx_watchdog.py` that monitors stream health and signals the orchestrator (avatar ready, cam1 connected/dropped, restart commands)
- `ffmpeg/` — FFmpeg binaries for audio/video processing
- `rhubarb/` — Rhubarb Lip Sync for avatar mouth animation
- `experiment_metric/` — Benchmarking tools for reranker performance and boxgate gesture segmentation

## Commands

### Running the Full System
```bash
# Activate root venv, then:
python main_orchestrator.py
```
Or use `main.bat`. Alternative batch files: `no_gs.bat` (no gesture system, uses `dummy_gesture_control.py`), `no_mic.bat` (no mic, uses `dummy_no_mic.py` for text-only input).

### Setting Up a Microservice
Each Python microservice directory (`Chatbot_Phi2`, `Gesture_System`, `RAG`, `STT`, `TTS`) has its own venv and `requirements.txt`:
```bash
cd <ServiceDir>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### React Frontend
```bash
cd react_avatar
npm install
npm run dev -- --host
```

### Root Orchestrator Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # fastapi, uvicorn, requests, pydantic
```

## Key Design Decisions

- **Microservice isolation**: Each AI service runs in its own Python venv with independent dependencies. The orchestrator launches them as separate console processes (`cmd.exe /k` with venv activation).
- **Stream-gated startup**: AI services only launch after MediaMTX watchdog confirms both avatar and camera streams are stable, preventing wasted GPU resources.
- **RAG follow-up tracking**: When the product carousel is active, `current_focus_asin` is sent to RAG so follow-up questions are contextual to the product the user is viewing.
- **Gesture vocabulary**: swipe_left, swipe_right, swipe_up, swipe_down, grab, expand — mapped to carousel navigation and 3D product inspection.
- **Windows-only**: Process management uses `taskkill`, `CREATE_NO_WINDOW`, `CREATE_NEW_CONSOLE` — the system targets Windows deployment.

## External Dependencies (not in repo)

Must be downloaded separately and placed in the root:
- **FFmpeg** → `ffmpeg/`
- **Rhubarb Lip Sync** → `rhubarb/`
- **MediaMTX binary** → `mediamtx/`
- **Model weights and datasets** from Hugging Face → `Chatbot_Phi2/models/`, `Gesture_System/checkpoints/`, `react_avatar/public/` assets
