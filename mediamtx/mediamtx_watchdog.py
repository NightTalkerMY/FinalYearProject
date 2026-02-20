import subprocess
import time
import re
import requests
import sys
import os

# --- CONFIG ---
MEDIAMTX_CMD = "mediamtx.exe"
PI_RESTART_URL = "http://100.80.70.120:8000/restart"
STARTUP_WINDOW = 10.0

# --- REGEX ---
SESSION_RE = re.compile(r"\[session ([0-9a-f]+)\]")
PUBLISH_CAM1_RE = re.compile(r"is publishing to path 'cam1'")
PUBLISH_AVATAR_RE = re.compile(r"is publishing to path 'avatar', (\d+) track")
LOSS_RE = re.compile(r"RTP packets lost")

def main():
    # Assume we are running in the correct cwd or mediamtx is in PATH
    print(f"[WATCHDOG] Starting MediaMTX: {MEDIAMTX_CMD}")
    
    # Start MediaMTX
    proc = subprocess.Popen(
        MEDIAMTX_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )

    active_sessions = {} # session_id -> start_time

    try:
        for raw_line in iter(proc.stdout.readline, b""):
            try:
                line = raw_line.decode(errors="ignore").rstrip()
            except UnicodeDecodeError:
                continue

            # 1. Echo the log so the Orchestrator sees it
            print(line)
            sys.stdout.flush() # Force flush so Orchestrator gets it immediately

            # Extract session ID
            m_sess = SESSION_RE.search(line)
            sess_id = m_sess.group(1) if m_sess else None

            # --- LOGIC A: PI CAMERA (CAM1) ---
            if sess_id and PUBLISH_CAM1_RE.search(line):
                active_sessions[sess_id] = time.time()
                print(f"[WATCHDOG_STATUS] CAM1_CONNECTED (Session {sess_id})")
                sys.stdout.flush()
                continue

            if sess_id and LOSS_RE.search(line):
                start_time = active_sessions.get(sess_id)
                if start_time:
                    elapsed = time.time() - start_time
                    if elapsed <= STARTUP_WINDOW:
                        print(f"[WATCHDOG_LOG] Early RTP loss on cam1 ({elapsed:.1f}s). Requesting Pi Restart...")
                        try:
                            requests.post(PI_RESTART_URL, timeout=3)
                        except Exception as e:
                            print(f"[WATCHDOG_LOG] Failed to contact Pi: {e}")
                        active_sessions.pop(sess_id, None)
                    else:
                        pass # Late loss, ignore

            # --- LOGIC B: REACT AVATAR (AVATAR) ---
            # Look for: is publishing to path 'avatar', X track(s)
            av_match = PUBLISH_AVATAR_RE.search(line)
            if av_match:
                tracks = int(av_match.group(1))
                if tracks < 2:
                    # BUG DETECTED: Tell Orchestrator to restart React
                    print("[WATCHDOG_CMD] RESTART_AVATAR")
                    sys.stdout.flush()
                else:
                    # SUCCESS: Tell Orchestrator to proceed
                    print("[WATCHDOG_CMD] AVATAR_READY")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        print("[WATCHDOG] Stopping MediaMTX...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    main()