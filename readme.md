# Intelligent Holographic AI for Retail
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LLM Phi-2](https://img.shields.io/badge/LLM-Microsoft_Phi--2-0078D4?logo=microsoft&logoColor=white)
![Vector DB ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-E91E63?logo=databricks&logoColor=white)
![Vision ResNet](https://img.shields.io/badge/Computer_Vision-ResNet-FF6F00)
![STT Whisper](https://img.shields.io/badge/STT-OpenAI_Whisper-412991?logo=openai&logoColor=white)
![TTS ZipVoice](https://img.shields.io/badge/TTS-ZipVoice-17E29A)
![ASD](https://img.shields.io/badge/ASD-Teacher--Student_CNN-FF4081)
![Frontend React](https://img.shields.io/badge/Avatar-React-61DAFB?logo=react&logoColor=black)
![Media FFmpeg](https://img.shields.io/badge/Streaming-FFmpeg_%7C_MediaMTX-007808?logo=ffmpeg&logoColor=white)

This repository contains the core orchestration and microservices for an interactive, AI-powered holographic retail assistant. The system utilizes a distributed microservice architecture, integrating large language models, retrieval-augmented generation, dynamic gesture control, speech processing, active speaker detection, and a 3D React-based avatar — all coordinated by a central FastAPI orchestrator.

## 🎥 Product Demonstration

<p align="center">
  <a href="https://raw.githubusercontent.com/NightTalkerMY/FinalYearProject/main/assets/demo.mp4">
    <img src="assets/demo-preview.gif" alt="Holographic AI retail assistant demo (preview at 2x speed)" width="300">
  </a>
</p>

<p align="center">
  🔇 <em>Preview above is muted &amp; 2× speed</em> — ▶️ <strong><a href="https://raw.githubusercontent.com/NightTalkerMY/FinalYearProject/main/assets/demo.mp4">watch the full demo with audio</a></strong> (56s, MP4)
</p>

The user speaks and gestures at the hologram: speech is transcribed, routed through RAG for product knowledge, answered by the LLM, and spoken back by the lip-synced 3D avatar — while hand gestures drive the product carousel and 3D product inspection in real time.

## 🌟 Key Innovations & Contributions

While the foundational architecture builds upon established research, this project introduces system-level optimizations to satisfy the latency, accuracy, and responsiveness constraints of a real-time retail deployment:

### RAG & LLM Pipeline Enhancements
* **Length-Aware Reranking:** The cross-encoder reranking stage was optimized by introducing length-aware document arrangement prior to inference. This design minimizes padding inefficiencies, reducing overall inference latency while preserving retrieval quality. Performance was benchmarked against MS MARCO and custom retail datasets, maintaining strong Mean Reciprocal Rank (MRR) and Hit Rate metrics.
* **Instruction-Tuned Semantic Routing:** Traditional precomputed query matching was replaced with a dynamic, instruction-tuned semantic routing mechanism. Incoming queries are encoded using a task-specific instruction function Φ with an instruction prefix (I_task) and compared directly against raw document embeddings. Evaluation on retail datasets showed measurable improvements in macro recall, F1 score, and precision, enabling more adaptive and context-aware retrieval.
* **Contextual Follow-Up Tracking:** When the product carousel is active, the ASIN of the product currently in focus is forwarded to the RAG service, so follow-up questions ("how much is *this* one?") are answered in the context of the product the user is actually looking at.

### Dynamic Gesture Control Enhancements
* **Real-Time Boxgate Logic:** The baseline gesture capture pipeline was re-architected from a manual, keyboard-triggered termination model to a fully automated, continuous inference loop using custom boxgate logic. This enables real-time segmentation without user intervention.
* **Performance Optimization:** By eliminating manual termination overhead, the system achieves higher gesture segmentation purity and lower latency variance, resulting in smoother interaction and improved perceptual continuity for the holographic avatar.
* **Gesture Vocabulary:** `swipe_left`, `swipe_right`, `swipe_up`, `swipe_down`, `grab`, and `expand` — mapped to carousel navigation and 3D product inspection.

### Real-Time Active Speaker Detection (`online_ASD/`)
* **Teacher-Student Distillation:** Seven specialized CNN "teacher" encoders (face, body, and background cues) are distilled into a lightweight `CausalStudentASD` model, enabling real-time speaker detection on edge hardware (Raspberry Pi).
* **Causal Online Pipeline:** S3FD face detection, IoU-based tracking, and a 50-frame sliding window with causal attention let the system classify each tracked face as speaking or silent live, so the avatar knows who is addressing it. See [`online_ASD/CLAUDE.md`](online_ASD/CLAUDE.md) for full details.

> 📊 **Detailed Evaluation & Metrics**
> The benchmarking tools live in [`experiment_metric/`](experiment_metric/): `reranker_metric/` evaluates retrieval quality (MS MARCO + a 100-question retail QnA set) and `boxgate_metric/` evaluates gesture segmentation performance.

## 🏗️ System Architecture & Microservices

The central orchestrator (`main_orchestrator.py`, FastAPI on port **5000**) launches and supervises every component. AI services are started only after the MediaMTX watchdog confirms both the avatar stream and the Pi camera stream are stable, and the full voice pipeline runs **STT → RAG → LLM → TTS → lip sync**.

Each AI service is an independent microservice with its own virtual environment and `main.py`:

| Service | Directory | Port | Tech |
|---------|-----------|------|------|
| Orchestrator | `main_orchestrator.py` (root) | 5000 | FastAPI |
| STT | `STT/` | 8000 | OpenAI Whisper |
| LLM | `Chatbot_Phi2/` | 8001 | Microsoft Phi-2 (fine-tuned) |
| RAG | `RAG/` | 8002 | ChromaDB + instruction-tuned semantic routing |
| TTS | `TTS/ZipVoice/` | 8003 | ZipVoice |
| Lip Sync | `TTS/allosaurus/` | 8004 | Allosaurus viseme server |
| Gesture | `Gesture_System/real-time-HGR-application/` | 8889 | ResNet hand gesture recognition |

Supporting components:

* **`react_avatar/`**: Frontend 3D avatar built with React + Vite. `launch-hologram.js` drives the Puppeteer-based hologram display, streamed via MediaMTX + FFmpeg.
* **`online_ASD/`**: Real-time active speaker detection subsystem (teacher-student CNN architecture).
* **`raspi/`**: Raspberry Pi clients — `camera/` streams the Pi camera feed to the server (with its own watchdog), `mic/` handles microphone capture.
* **`mediamtx/`**: MediaMTX configuration plus `mediamtx_watchdog.py`, which monitors stream health and signals the orchestrator (avatar ready, camera connected/dropped, restart commands).
* **`experiment_metric/`**: Benchmarking tools for reranker performance and boxgate gesture segmentation.

## 📥 Prerequisites & External Dependencies

Before running the system, several external binaries and large model assets must be downloaded.

> ⚠️ **Windows-only deployment:** the orchestrator's process management (`taskkill`, console creation flags) targets Windows.

### 1. External Binaries
Download the following tools and place them in the root directory (or respective folder):
* **FFmpeg:** Required for audio/video processing. Download from https://github.com/BtbN/FFmpeg-Builds/releases and find latest assets named `ffmpeg-master-latest-win64-gpl-shared.zip`. Then extract to the root `ffmpeg/` directory.
* **Rhubarb Lip Sync:** Required for avatar lip-sync generation. Download from https://github.com/DanielSWolf/rhubarb-lip-sync/releases/tag/v1.14.0 and find the latest assets named `Rhubarb-Lip-Sync-1.14.0-Windows.zip`. Then extract to the root `rhubarb/` directory.
* **MediaMTX:** Required for media streaming. Download the binary from https://github.com/bluenviron/mediamtx/releases/tag/v1.16.1 and find the latest assets named `mediamtx_v1.16.1_windows_amd64.zip`. Then place it inside the `mediamtx/` directory alongside the configuration files.

### 2. Hugging Face Assets (Models, Datasets & 3D Files)
Due to file size limits, datasets, fine-tuned models, and heavy 3D assets are hosted externally on Hugging Face: **[INSERT_HUGGINGFACE_PROFILE_LINK]**

Please download and place the following assets into their respective directories:
* **`Chatbot_Phi2/`**: Download the specific datasets and model weights.
* **`Gesture_System/`**: Download the ResNet training datasets and inference models.
* **`react_avatar/`**: Download the `public/` directory containing the rendered 3D avatar files and place it inside the frontend folder.
* **`online_ASD/`**: Download the pretrained teacher checkpoints and distilled student weights into `pretrain_model/`.

## ⚙️ Installation & Setup

Because this project uses a microservice architecture, **each Python directory requires its own separate virtual environment**.

### Step 1: Setup Python Microservices
For each of the following directories (`Chatbot_Phi2`, `Gesture_System`, `RAG`, `STT`, `TTS`, `online_ASD`), navigate into the folder, create a virtual environment, and install its specific dependencies:

```bash
cd [Directory_Name]
python -m venv venv

# Activate the venv (Windows):
venv\Scripts\activate
# OR Activate the venv (Mac/Linux):
source venv/bin/activate

pip install -r requirements.txt
deactivate
cd ..
```

### Step 2: Setup the React Avatar

Navigate to the frontend directory and install the Node packages:

```bash
cd react_avatar
npm install
cd ..
```

### Step 3: Setup the Main Orchestrator

Finally, setup the root environment that ties everything together:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Running the System

The entire microservice architecture is fully automated through the central orchestrator. You do not need to manually start each individual component — the orchestrator launches the MediaMTX watchdog, the React frontend, and every AI microservice, gating AI startup on stream stability to avoid wasting GPU resources.

To launch the complete Intelligent Holographic AI system:

1. Open your terminal in the root directory.
2. Ensure your root virtual environment is activated.
3. Run the orchestrator:

```bash
python main_orchestrator.py
```

Or use one of the provided batch launchers:

| Launcher | Purpose |
|----------|---------|
| `main.bat` | Full system |
| `no_gs.bat` | No gesture hardware — substitutes `dummy_gesture_control.py` |
| `no_mic.bat` | No microphone — substitutes `dummy_no_mic.py` for text-only input |

## 📚 Acknowledgements & References

This project builds upon and significantly modifies concepts from the following academic research:

* **RAG & LLM Architecture:** The foundational retrieval-augmented generation structure was inspired by *TeleOracle: Fine-Tuned Retrieval-Augmented Generation With Long-Context Support for Networks* (Alabbasi et al., IEEE Internet of Things Journal, 2025). In this repository, the architecture has been uniquely adapted and improved to support real-time retail microservices using Microsoft Phi-2 and ChromaDB.
* **Dynamic Gesture System:** The core vision methodology is based on *Skeleton-Based Real-Time Hand Gesture Recognition Using Data Fusion and Ensemble Multi-Stream CNN Architecture* (Habib, Yusuf, & Moustafa, MDPI Technologies, 2025). The system has been modified and fine-tuned for specialized, real-time interactive avatar control using ResNet.
