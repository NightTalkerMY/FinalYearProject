# Intelligent Holographic AI for Retail
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LLM Phi-2](https://img.shields.io/badge/LLM-Microsoft_Phi--2-0078D4?logo=microsoft&logoColor=white)
![Vector DB ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-E91E63?logo=databricks&logoColor=white)
![Vision ResNet](https://img.shields.io/badge/Computer_Vision-ResNet-FF6F00)
![STT Whisper](https://img.shields.io/badge/STT-OpenAI_Whisper-412991?logo=openai&logoColor=white)
![TTS Coqui](https://img.shields.io/badge/TTS-Coqui-17E29A)
![Frontend React](https://img.shields.io/badge/Avatar-React-61DAFB?logo=react&logoColor=black)
![Media FFmpeg](https://img.shields.io/badge/Streaming-FFmpeg_%7C_MediaMTX-007808?logo=ffmpeg&logoColor=white)

This repository contains the core orchestration and microservices for an interactive, AI-powered holographic retail assistant. The system utilizes a distributed microservice architecture, integrating large language models, retrieval-augmented generation, dynamic gesture control, speech processing, and a 3D React-based avatar.

## 🎥 Product Demonstration

*Video demonstration of the Intelligent Holographic AI system in action to be uploaded soon!*

## 🌟 Key Innovations & Contributions

While the foundational architecture builds upon existing research, this project introduces several novel optimizations to meet the strict latency and accuracy requirements of a real-time retail environment:

### RAG & LLM Pipeline Enhancements
* **Length-Aware Reranking:** Optimized the cross-encoder reranker by implementing length-aware document arrangement prior to processing. This significantly reduces padding waste and lowers inference latency, all while maintaining strict Mean Reciprocal Rank (MRR) and Hit Rate metrics (benchmarked against MS MARCO and custom retail datasets).
* **Instruction-Tuned Semantic Routing:** Replaced traditional precomputed query matching with direct document-embedding comparisons. By introducing a task-specific instruction function, $\Phi$, incoming queries are encoded with an instruction prefix ($I_{task}$) and compared directly against raw document embeddings. Evaluated against retail datasets, this dynamic routing approach demonstrated measurable improvements in macro recall, F1 score, and overall precision.

### Dynamic Gesture Control Enhancements
* **Real-Time Boxgate Logic:** Upgraded the baseline gesture recording phase from a manual, keyboard-triggered stop mechanism to a fully automated, real-time continuous inference loop utilizing custom boxgate logic.
* **Performance Optimization:** This shift from manual to automated triggering drastically reduces termination overhead and significantly increases gesture segmentation purity, ensuring a seamless user experience with the holographic avatar.

> 📊 **Detailed Evaluation & Metrics**
> For a comprehensive breakdown of the empirical data supporting these improvements—including MS MARCO benchmarks, retail dataset F1/precision scores, and latency tests—please refer to the `experiment_metric.md` file *(coming soon)*.

## 🏗️ System Architecture & Microservices

The project is divided into specialized directories. Each acts as an independent microservice with its own virtual environment and dependencies, all communicating with the central `main_orchestrator.py`.

* **`Chatbot_Phi2/`**: Core LLM engine directory. Contains code for fine-tuning and real-time inference, running as an independent `main.py` microservice.
* **`Gesture_System/`**: Dynamic hand gesture control system utilizing ResNet. Handles both model training and real-time vision inference via its own `main.py`.
* **`RAG/`**: Retrieval-Augmented Generation pipeline using ChromaDB for contextual memory and knowledge retrieval.
* **`STT/`**: Speech-to-Text voice transcription layer powered by OpenAI Whisper.
* **`TTS/`**: Text-to-Speech voice generation layer using Coqui TTS.
* **`react_avatar/`**: Frontend 3D avatar rendering layer built with React.
* **`mediamtx/`**: Contains the configuration files for real-time media routing and streaming.

## 📥 Prerequisites & External Dependencies

Before running the system, several external binaries and large model assets must be downloaded. 

### 1. External Binaries
Download the following tools and place them in the root directory (or respective folder):
* **FFmpeg:** Required for audio/video processing. Download from https://github.com/BtbN/FFmpeg-Builds/releases and find latest assests named `ffmpeg-master-latest-win64-gpl-shared.zip`.Then extract to the root `ffmpeg/` directory.
* **Rhubarb Lip Sync:** Required for avatar lip-sync generation. Download from https://github.com/DanielSWolf/rhubarb-lip-sync/releases/tag/v1.14.0 and find the latest assests named `Rhubarb-Lip-Sync-1.14.0-Windows.zip` . Then extract to the root `rhubarb/` directory.
* **MediaMTX:** Required for media streaming. Download the binary from https://github.com/bluenviron/mediamtx/releases/tag/v1.16.1 and find the latest assests named `mediamtx_v1.16.1_windows_amd64.zip` . Then place it inside the `mediamtx/` directory alongside the configuration files.

### 2. Hugging Face Assets (Models, Datasets & 3D Files)
Due to file size limits, datasets, fine-tuned models, and heavy 3D assets are hosted externally on Hugging Face: **[INSERT_HUGGINGFACE_PROFILE_LINK]**

Please download and place the following assets into their respective directories:
* **`Chatbot_Phi2/`**: Download the specific datasets and model weights.
* **`Gesture_System/`**: Download the ResNet training datasets and inference models.
* **`react_avatar/`**: Download the `public/` directory containing the rendered 3D avatar files and place it inside the frontend folder.

## ⚙️ Installation & Setup

Because this project uses a microservice architecture, **each Python directory requires its own separate virtual environment**. 

### Step 1: Setup Python Microservices
For each of the following directories (`Chatbot_Phi2`, `Gesture_System`, `RAG`, `STT`, `TTS`), navigate into the folder, create a virtual environment, and install its specific dependencies:

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

The entire microservice architecture is fully automated through the central orchestrator. You do not need to manually start each individual component.

To launch the complete Intelligent Holographic AI system:

1. Open your terminal in the root directory.
2. Ensure your root virtual environment is activated.
3. Run the orchestrator:

```bash
python main_orchestrator.py


```

*(Note: `dummy_gesture_control.py` and `dummy_no_mic.py` are provided at the root level for testing isolated orchestrator components without full hardware requirements).*

## 📚 Acknowledgements & References

This project builds upon and significantly modifies concepts from the following academic research:

* **RAG & LLM Architecture:** The foundational retrieval-augmented generation structure was inspired by *TeleOracle: Fine-Tuned Retrieval-Augmented Generation With Long-Context Support for Networks* (Alabbasi et al., IEEE Internet of Things Journal, 2025). In this repository, the architecture has been uniquely adapted and improved to support real-time retail microservices using Microsoft Phi-2 and ChromaDB.
* **Dynamic Gesture System:** The core vision methodology is based on *Skeleton-Based Real-Time Hand Gesture Recognition Using Data Fusion and Ensemble Multi-Stream CNN Architecture* (Habib, Yusuf, & Moustafa, MDPI Technologies, 2025). The system has been modified and fine-tuned for specialized, real-time interactive avatar control using ResNet.
