import asyncio
import websockets
import json

connected_clients = set()

async def handler(websocket):  # ← only one arg now
    connected_clients.add(websocket)
    print("Client connected")
    try:
        async for message in websocket:
            print(f"Received from client: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected_clients.remove(websocket)

async def wait_for_input():
    while True:
        user_input = await asyncio.to_thread(input, "Type 'yes' to trigger lipsync: ")
        if user_input.strip().lower() == "yes":
            message = json.dumps({"play": True})
            # iterate over a copy so we don't mutate while iterating
            for ws in list(connected_clients):
                try:
                    await ws.send(message)
                except websockets.exceptions.ConnectionClosed:
                    connected_clients.discard(ws)
            print("Trigger sent.")
        else:
            print("Invalid input. Type 'yes' to trigger.")

async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8000)
    print("WebSocket server running at ws://localhost:8000")
    await wait_for_input()  # keeps the program alive

if __name__ == "__main__":
    asyncio.run(main())
