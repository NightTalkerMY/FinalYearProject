# dummy_post_interactive.py

import requests
import time
from pathlib import Path

URL = "http://127.0.0.1:8003/generate_speech"

def wait_for_viseme(json_path: Path, timeout=5.0, interval=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if json_path.exists():
            print(f"[VISEME OK] {json_path}")
            return True
        time.sleep(interval)
    print("[VISEME TIMEOUT]")
    return False


while True:
    text = input("\nInput text (or /exit): ").strip()
    if text.lower() in ["/exit", "exit", "quit", "/q"]:
        break

    if not text:
        continue

    resp = requests.post(URL, json={"text": text})
    if resp.status_code != 200:
        print("[CLIENT] Error:", resp.status_code, resp.text)
        continue

    data = resp.json()
    print("[CLIENT] Response:", data)

    audio_path = Path(data["audio_path"])
    viseme_path = Path(data["viseme_path"])

    print("[CLIENT] Waiting for viseme...")
    wait_for_viseme(viseme_path)
