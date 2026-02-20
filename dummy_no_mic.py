import requests

# The new endpoint we just created
URL = "http://127.0.0.1:5000/process_text"

print("SILENT MODE ACTIVATED")
print("Type your command and press ENTER. (Type 'exit' to quit)")
print("-" * 40)

while True:
    try:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input.strip():
            continue

        print("... Sending to Orchestrator ...")
        
        try:
            # Send as JSON text
            res = requests.post(URL, json={"text": user_input})
            
            if res.status_code == 200:
                data = res.json()
                print(f"AI: {data.get('text', 'No text response')}")
            else:
                print(f"Error {res.status_code}: {res.text}")
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to Orchestrator. Is it running on port 5000?")

    except KeyboardInterrupt:
        print("\nExiting...")
        break