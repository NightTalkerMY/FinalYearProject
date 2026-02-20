import subprocess
import time
import requests
import uvicorn
import threading
import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ==========================================
# CONFIGURATION & PATH AUTO-DETECT
# ==========================================

# 1. Define the Project Root (Where this script is running)
PROJECT_ROOT = Path(__file__).parent.resolve()

# 2. Define expected paths
MEDIAMTX_DIR = Path(r"C:\Users\wwon0076\Desktop\FYP\mediamtx")
WATCHDOG_PYTHON = MEDIAMTX_DIR / "venv" / "Scripts" / "python.exe"
WATCHDOG_SCRIPT = "mediamtx_watchdog.py"

# 3. Auto-Detect React/Node Directory
# Prioritize 'react_avatar', fallback to 'frontend'
if (PROJECT_ROOT / "react_avatar").is_dir():
    REACT_DIR = PROJECT_ROOT / "react_avatar"
    NODE_SCRIPT = REACT_DIR / "launch-hologram.js" # User specified hyphen
    print(f"Detected 'react_avatar' directory. Using: {REACT_DIR}")
elif (PROJECT_ROOT / "frontend").is_dir():
    REACT_DIR = PROJECT_ROOT / "frontend"
    NODE_SCRIPT = REACT_DIR / "launch_hologram.js" # Standard underscore
    print(f"Detected 'frontend' directory. Using: {REACT_DIR}")
else:
    REACT_DIR = PROJECT_ROOT
    NODE_SCRIPT = PROJECT_ROOT / "launch_hologram.js"
    print(f"frontend' directory not found. Using Root: {REACT_DIR}")

PATHS = {
    "MEDIAMTX_DIR": str(MEDIAMTX_DIR),
    "WATCHDOG_PYTHON": str(WATCHDOG_PYTHON),
    "WATCHDOG_SCRIPT": WATCHDOG_SCRIPT,
    "REACT_DIR": str(REACT_DIR),
    "NODE_SCRIPT": str(NODE_SCRIPT),
}

# --- AI SERVICES (Gesture Included) ---
AI_SERVICES = {
    "STT": {"dir": "STT", "url": "http://127.0.0.1:8000/transcribe"},
    "LLM": {"dir": "Chatbot_Phi2", "url": "http://127.0.0.1:8001/chat"},
    "RAG": {"dir": "RAG", "url": "http://127.0.0.1:8002/get_context"},
    "TTS": {"dir": "TTS", "url": "http://127.0.0.1:8003/generate_speech"},
    "GESTURE": {
        "dir": "Gesture_System/real-time-HGR-application", 
        "venv": "..\\venv", 
        "url": "http://127.0.0.1:8889"
    } 
}

# ==========================================
# APP SETUP
# ==========================================
app = FastAPI(title="PUMA Holographic Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_OUTPUT_DIR = Path("./TTS/outputs_xtts").resolve()
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=TTS_OUTPUT_DIR), name="audio")

# Global State
SYSTEM_STATE = {
    "status": "BOOTING",
    "audio_url": None,
    "viseme_url": None,
    "trigger_carousel": False,
    "gesture": "talk",
    "asins": [],
    "last_update_id": 0,
    "streams": {"avatar": False, "cam1": False},
    "ai_launched": False
}

PROCS = {
    "watchdog": None,
    "react": None,
    "node": None,
    "ai_services": []
}

# ==========================================
# PROCESS MANAGEMENT
# ==========================================

def kill_process_tree(proc):
    """Forcefully kills a process and its children on Windows."""
    if proc and proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error killing process: {e}")

def validate_paths():
    """Checks if critical files exist before launching."""
    print("Validating Paths...")
    missing = False
    
    if not os.path.exists(PATHS["MEDIAMTX_DIR"]):
        print(f"Missing MediaMTX Dir: {PATHS['MEDIAMTX_DIR']}")
        missing = True
    
    if not os.path.exists(PATHS["REACT_DIR"]):
        print(f"Missing React Dir: {PATHS['REACT_DIR']}")
        missing = True

    if not os.path.exists(PATHS["NODE_SCRIPT"]):
        print(f"Missing Node Script: {PATHS['NODE_SCRIPT']}")
        # Fallback check for alternate spelling
        alt_script = Path(PATHS["REACT_DIR"]) / "launch_hologram.js"
        if alt_script.exists():
             print(f"Found script at {alt_script}, updating path...")
             PATHS["NODE_SCRIPT"] = str(alt_script)
        else:
             missing = True
            
    if missing:
        print("STOPPING: Please fix paths in main_orchestrator.py")
        sys.exit(1)
    print("Paths Verified.")

def start_react_stack():
    """Starts npm run dev and the node hologram launcher."""
    print(f"\n[ORCHESTRATOR] Launching Frontend Stack from {PATHS['REACT_DIR']}...")
    
    # 1. React (Vite) - HIDDEN
    PROCS["react"] = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host"], 
        cwd=PATHS["REACT_DIR"],
        shell=True, 
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    
    # 2. Node Launcher - HIDDEN
    PROCS["node"] = subprocess.Popen(
        ["node", PATHS["NODE_SCRIPT"]],
        cwd=os.path.dirname(PATHS["NODE_SCRIPT"]),
        creationflags=subprocess.CREATE_NO_WINDOW
    )

def restart_react_stack():
    """Restarts React/Node when Watchdog detects the 1-track bug."""
    print("\n[ORCHESTRATOR] 1-Track Bug Detected. Restarting Frontend...")
    kill_process_tree(PROCS["react"])
    kill_process_tree(PROCS["node"])
    time.sleep(2) # Wait for ports to clear
    start_react_stack()

def launch_ai_services():
    """Starts the Python AI backend services ONLY if both streams are ready."""
    if SYSTEM_STATE["ai_launched"]: return
    
    # --- STRICT CHECK: AVATAR AND CAM1 MUST BE READY ---
    if not SYSTEM_STATE["streams"]["avatar"]:
        print("Waiting for Avatar Stream...")
        return
    if not SYSTEM_STATE["streams"]["cam1"]:
        print("Waiting for Pi Camera Stream...")
        return

    print("\n[ORCHESTRATOR] All Streams Stable (Avatar + Cam1). Launching AI Services...")
    for name, cfg in AI_SERVICES.items():
        print(f"   > Launching {name}...")
        venv_path = cfg.get("venv", ".\\venv")
        # Ensure we change directory to the service folder before activating venv
        cmd = f'cd {cfg["dir"]} && {venv_path}\\Scripts\\activate && python main.py'
        p = subprocess.Popen(["cmd.exe", "/k", cmd], creationflags=subprocess.CREATE_NEW_CONSOLE)
        PROCS["ai_services"].append(p)
    
    SYSTEM_STATE["ai_launched"] = True
    SYSTEM_STATE["status"] = "IDLE"
    print("[ORCHESTRATOR] All Systems Operational.\n")

# ==========================================
# WATCHDOG LISTENER THREAD
# ==========================================

def watchdog_listener():
    """Runs the watchdog script and listens for commands in stdout."""
    print(f"[ORCHESTRATOR] Starting Watchdog (MediaMTX Monitor)...")
    
    cmd = [PATHS["WATCHDOG_PYTHON"], PATHS["WATCHDOG_SCRIPT"]]
    
    if not Path(PATHS["WATCHDOG_PYTHON"]).exists():
        print(f"FATAL: Watchdog Python not found at {PATHS['WATCHDOG_PYTHON']}")
        return

    PROCS["watchdog"] = subprocess.Popen(
        cmd,
        cwd=PATHS["MEDIAMTX_DIR"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    while True:
        output = PROCS["watchdog"].stdout.readline()
        if output == b'' and PROCS["watchdog"].poll() is not None:
            break
        
        if output:
            try:
                line = output.decode().strip()
                if "[MTX_RAW]" in line:
                    pass
                else:
                    print(line) 
                
                # --- COMMAND HANDLING ---
                if "[WATCHDOG_CMD] RESTART_AVATAR" in line:
                    restart_react_stack()
                
                elif "[WATCHDOG_CMD] AVATAR_READY" in line:
                    if not SYSTEM_STATE["streams"]["avatar"]:
                        print("[ORCHESTRATOR] Avatar Verified (2 Tracks).")
                        SYSTEM_STATE["streams"]["avatar"] = True
                        launch_ai_services()
                
                elif "[WATCHDOG_STATUS] CAM1_CONNECTED" in line:
                    if not SYSTEM_STATE["streams"]["cam1"]:
                        print("[ORCHESTRATOR] Pi Camera Connected.")
                        SYSTEM_STATE["streams"]["cam1"] = True
                        launch_ai_services()

            except Exception:
                continue

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/process")
async def process_voice_command(request: Request):
    global SYSTEM_STATE
    audio_bytes = await request.body()
    print("\n--- [PIPELINE STARTED] ---")
    
    # 1. STT
    try:
        stt_res = requests.post(AI_SERVICES["STT"]["url"], data=audio_bytes).json()
        user_text = stt_res.get("text", "")
        print(f"User said: {user_text}")
    except Exception as e:
        print(f"STT Failed: {e}")
        return {"status": "error"}

    if not user_text: return {"status": "ok"}

    # 2. RAG
    print(f"Sending to RAG...")
    try:
        rag_res = requests.post(AI_SERVICES["RAG"]["url"], json={"query": user_text}).json()
        context = rag_res.get("context", "N/A")
        trigger_carousel = rag_res.get("trigger_carousel", False)
        asins = rag_res.get("asins", [])
        print(f"Context found. Carousel: {trigger_carousel}")
    except:
        context, trigger_carousel, asins = "N/A", False, []
    
    # 3. LLM
    print(f"Sending to LLM...")
    try:
        llm_res = requests.post(AI_SERVICES["LLM"]["url"], json={"context": context, "query": user_text}).json()
        response_text = llm_res.get("response", "")
    except:
        response_text = "I am having trouble thinking."

    # 4. TTS
    print(f"Sending to TTS...")
    try:
        tts_res = requests.post(AI_SERVICES["TTS"]["url"], json={"text": response_text}).json()
        filename = tts_res.get("filename")
        
        if filename:
            base = "http://localhost:5000/audio"
            
            # --- FIXED GESTURE LOGIC ---
            gesture = "talk"
            if trigger_carousel:
                gesture = "transition"
            elif "N/A" in str(context) or "No products found" in str(context):
                gesture = "confused"

            SYSTEM_STATE.update({
                "status": "SPEAKING",
                "audio_url": f"{base}/{filename}",
                "viseme_url": f"{base}/{filename.replace('.wav', '.json')}",
                "trigger_carousel": trigger_carousel,
                "asins": asins,
                "gesture": gesture,
                "last_update_id": SYSTEM_STATE["last_update_id"] + 1
            })
            print(f"Playing: {filename} (Gesture: {gesture})")
    except Exception as e:
        print(f"TTS Failed: {e}")

    print("--- [PIPELINE COMPLETE] ---\n")
    return {"status": "ok", "text": response_text}

@app.post("/process_text")
async def process_text_command(request: Request):
    global SYSTEM_STATE
    data = await request.json()
    user_text = data.get("text", "")
    print(f"\nUser Typed: {user_text}")
    if not user_text: return {"status": "empty"}

    # --- RESTORED FULL PIPELINE FOR TEXT ---
    
    # 1. RAG (Text)
    print(f"Sending to RAG...")
    try:
        rag_res = requests.post(AI_SERVICES["RAG"]["url"], json={"query": user_text}).json()
        context = rag_res.get("context", "N/A")
        trigger_carousel = rag_res.get("trigger_carousel", False)
        asins = rag_res.get("asins", [])
        print(f"Context found. Carousel: {trigger_carousel}")
    except:
        context, trigger_carousel, asins = "N/A", False, []

    # 2. LLM (Text)
    print(f"Sending to LLM...")
    try:
        llm_res = requests.post(AI_SERVICES["LLM"]["url"], json={"context": context, "query": user_text}).json()
        response_text = llm_res.get("response", "")
    except:
        response_text = "I am having trouble thinking."

    # 3. TTS (Text)
    print(f"Sending to TTS...")
    try:
        tts_res = requests.post(AI_SERVICES["TTS"]["url"], json={"text": response_text}).json()
        filename = tts_res.get("filename")
        base = "http://localhost:5000/audio"
        
        # --- FIXED GESTURE LOGIC (Text Mode) ---
        gesture = "talk"
        if trigger_carousel:
            gesture = "transition"
        elif "N/A" in str(context) or "No products found" in str(context):
            gesture = "confused"

        SYSTEM_STATE.update({
            "status": "SPEAKING",
            "audio_url": f"{base}/{filename}",
            "viseme_url": f"{base}/{filename.replace('.wav', '.json')}",
            "trigger_carousel": trigger_carousel,
            "asins": asins,
            "gesture": gesture,
            "last_update_id": SYSTEM_STATE["last_update_id"] + 1
        })
        print(f"Playing: {filename} (Gesture: {gesture})")
    except Exception as e: print(f"TTS Error: {e}")

    return {"status": "ok"}

@app.get("/poll_state")
def poll_state():
    return SYSTEM_STATE

@app.post("/reset_state")
def reset_state():
    SYSTEM_STATE["status"] = "IDLE"
    return {"status": "reset"}

@app.post("/generate_goodbye")
def generate_goodbye():
    print("REACT REQUESTED GOODBYE SPEECH")
    
    if not SYSTEM_STATE["trigger_carousel"]:
        print("Ignoring duplicate goodbye request.")
        return {"status": "ignored"}

    SYSTEM_STATE["trigger_carousel"] = False
    
    try:
        tts_res = requests.post(AI_SERVICES["TTS"]["url"], json={"text": "I hope you liked those. Let me know if you need anything else!"}).json()
        filename = tts_res.get("filename")
        ts = int(time.time())
        base = "http://localhost:5000/audio"
        
        SYSTEM_STATE.update({
            "status": "SPEAKING",
            "audio_url": f"{base}/{filename}?t={ts}",
            "viseme_url": f"{base}/{filename.replace('.wav', '.json')}?t={ts}",
            "gesture": "talk",
            "last_update_id": SYSTEM_STATE["last_update_id"] + 1
        })
    except Exception as e:
        print(f"TTS Error: {e}")
    
    return {"status": "goodbye_initiated"}

@app.post("/gesture_command")
async def handle_gesture(request: Request):
    data = await request.json()
    cmd = data.get("command")
    print(f"GESTURE RELAY: {cmd}")
    SYSTEM_STATE["gesture"] = cmd
    SYSTEM_STATE["last_update_id"] += 1
    return {"status": "relayed"}

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("   PUMA ORCHESTRATOR + WATCHDOG INTEGRATION       ")
    print("==================================================")

    # 0. Validate Paths
    validate_paths()

    # 1. Start Watchdog (MediaMTX)
    t = threading.Thread(target=watchdog_listener, daemon=True)
    t.start()

    # 2. Wait for MediaMTX to spin up
    print("Waiting 3s for MediaMTX to warm up...")
    time.sleep(3)

    # 3. Start Frontend (React + Node)
    # These windows are HIDDEN (CREATE_NO_WINDOW)
    start_react_stack()

    # 4. Start API Server
    print("Starting API Server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")
    except KeyboardInterrupt:
        print("\nSHUTTING DOWN...")
        kill_process_tree(PROCS["watchdog"])
        kill_process_tree(PROCS["react"])
        kill_process_tree(PROCS["node"])
        for p in PROCS["ai_services"]: kill_process_tree(p)