# pi_watchdog.py  (run this on the Pi instead of pi_image_to_desktop.py)

import asyncio
import subprocess
from aiohttp import web

# Change this to wherever your streaming script lives
STREAM_CMD = ["python3", "/home/william/Desktop/local_env/send_picamera_stream_to_server.py"]

class StreamManager:
    def __init__(self):
        self.proc = None

    async def start(self):
        if self.proc and self.proc.poll() is None:
            print("[PI] Stream already running")
            return
        print("[PI] Starting stream process:", " ".join(STREAM_CMD))
        # Start child process; don't capture stdout/stderr here unless you want it
        self.proc = subprocess.Popen(STREAM_CMD)

    async def stop(self):
        if not self.proc or self.proc.poll() is not None:
            print("[PI] Stream not running")
            return
        print("[PI] Stopping stream process...")
        self.proc.terminate()
        try:
            # Wait up to 5s for graceful shutdown
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.proc.wait)
        except Exception:
            print("[PI] Force killing stream process")
            self.proc.kill()
        finally:
            self.proc = None

    async def restart(self):
        print("[PI] Restart requested")
        await self.stop()
        await self.start()

manager = StreamManager()

# -------- HTTP handlers --------

async def handle_restart(request: web.Request):
    await manager.restart()
    return web.json_response({"status": "restarted"})

async def handle_status(request: web.Request):
    running = manager.proc is not None and manager.proc.poll() is None
    return web.json_response({"running": running})

async def on_startup(app: web.Application):
    # Automatically start the stream when watchdog launches
    await manager.start()

def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_post("/restart", handle_restart)
    app.router.add_get("/status", handle_status)

    # Listen on all interfaces so the desktop (over Tailscale) can reach it
    web.run_app(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
