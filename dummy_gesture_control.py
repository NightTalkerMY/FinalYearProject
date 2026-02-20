import requests

# Configuration
ORCHESTRATOR_URL = "http://127.0.0.1:5000/gesture_command"

print("=========================================")
print("     SIMPLE GESTURE TRIGGER       ")
print("=========================================")
print(" [l] Swipe Left   (Prev Item)")
print(" [r] Swipe Right  (Next Item)")
print(" [u] Swipe Up     (Rotate X+)")
print(" [d] Swipe Down   (Rotate X-)")
print(" [g] Grab         (Enter Inspection)")
print(" [e] Expand       (Exit / Back)")
print(" [q] Quit")
print("=========================================")

def send_gesture(command):
    payload = {"command": command, "confidence": 1.0}
    try:
        response = requests.post(ORCHESTRATOR_URL, json=payload, timeout=0.5)
        if response.status_code == 200:
            print(f"Sent: {command}")
        else:
            print(f"Server Error: {response.status_code}")
    except Exception as e:
        print(f"Connection Failed. Is Orchestrator running?")

while True:
    # Simple standard input - no fancy library needed
    user_input = input("\nEnter key (l/r/u/d/g/e): ").strip().lower()

    if user_input == 'l':
        send_gesture("swipe_left")
    elif user_input == 'r':
        send_gesture("swipe_right")
    elif user_input == 'u':
        send_gesture("swipe_up")
    elif user_input == 'd':
        send_gesture("swipe_down")
    elif user_input == 'g':
        send_gesture("grab")
    elif user_input == 'e':
        send_gesture("expand")
    elif user_input == 'q':
        print("Bye!")
        break
    else:
        print("Invalid key. Use l, r, u, d, g, or e.")