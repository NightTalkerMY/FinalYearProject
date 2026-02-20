# import torch
# import re
# from fastapi import FastAPI, Request # Only standard FastAPI
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import uvicorn

# BASE_MODEL_ID = "microsoft/phi-2"
# TUNED_MODEL_PATH = "models/phi2_retail_native_bf16_c6e0c0"

# tokenizer = AutoTokenizer.from_pretrained(TUNED_MODEL_PATH, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_ID,
#     dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )
# model = PeftModel.from_pretrained(base_model, TUNED_MODEL_PATH)
# model.eval()

# SYSTEM_PROMPT = (
#     "You are the PUMA Holographic Assistant. Follow these strict operational rules:\n"
#     "1. If Context is 'N/A': Handle general greetings or PUMA-related brand questions. "
#     "If the query is completely unrelated to PUMA, sports, or retail, politely refuse to answer.\n"
#     "2. If Context is 'No products found.': Inform the user that no matching footwear was found "
#     "and suggest they try a different style or category.\n"
#     "3. If Context contains Product Lists: Provide a high-level highlight of the collection "
#     "and transition the user into the immersive 3D view.\n"
#     "4. If Context contains T&C/Policies: Use the information provided to answer the user query accurately.\n"
#     "5. If User Query is '<GESTURE_EXIT>': Acknowledge that the user has closed the 3D display, "
#     "briefly summarize the product they just viewed, and ask if they need further assistance."
# )

# app = FastAPI()

# @app.post("/chat")
# async def chat(request: Request):
#     data = await request.json() 
#     context = data.get("context", "N/A")
#     query = data.get("query", "")

#     # --- YOUR ORIGINAL INFERENCE LOGIC ---
#     prompt = (
#         f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
#         f"### Context:\n{context}\n\n"
#         f"### User Query:\n{query}\n\n"
#         f"### Response:\n"
#     )

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

#     raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     response_text = raw_output.split("### Response:\n", 1)[-1]
#     response_text = re.sub(r"<END_OF_RESPONSE.*", "", response_text, flags=re.DOTALL).strip()

#     return {"response": response_text}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001)


#### Fixed Inferencing Time ####
import torch
import re
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import uvicorn

BASE_MODEL_ID = "microsoft/phi-2"
TUNED_MODEL_PATH = "models/phi2_retail_native_bf16_c6e0c0"

# -----------------------------
# Stopper: stop at <END_OF_RESPONSE>
# -----------------------------
class StopOnTokens(StoppingCriteria):
    """Stop generation when a specific token sequence appears at the end."""
    def __init__(self, stop_ids: list[int]):
        if not stop_ids:
            raise ValueError("stop_ids is empty")
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[0] != 1:
            return False
        n = len(self.stop_ids)
        if input_ids.shape[1] < n:
            return False
        return input_ids[0, -n:].tolist() == self.stop_ids


# -----------------------------
# Model load
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(TUNED_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, TUNED_MODEL_PATH)
model.eval()

# Precompute stop ids once (fast)
STOP_STR = "<END_OF_RESPONSE>"  # your finetuned end tag
STOP_IDS = tokenizer.encode(STOP_STR, add_special_tokens=False)
STOPPING = StoppingCriteriaList([StopOnTokens(STOP_IDS)])

SYSTEM_PROMPT = (
    "You are the PUMA Holographic Assistant. Follow these strict operational rules:\n"
    "1. If Context is 'N/A': Handle general greetings or PUMA-related brand questions. "
    "If the query is completely unrelated to PUMA, sports, or retail, politely refuse to answer.\n"
    "2. If Context is 'No products found.': Inform the user that no matching footwear was found "
    "and suggest they try a different style or category.\n"
    "3. If Context contains Product Lists: Provide a high-level highlight of the collection "
    "and transition the user into the immersive 3D view.\n"
    "4. If Context contains T&C/Policies: Use the information provided to answer the user query accurately.\n"
    "5. If User Query is '<GESTURE_EXIT>': Acknowledge that the user has closed the 3D display, "
    "briefly summarize the product they just viewed, and ask if they need further assistance.\n\n"
)

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    context = data.get("context", "N/A")
    query = data.get("query", "")

    prompt = (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Context:\n{context}\n\n"
        f"### User Query:\n{query}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,                 # safe ceiling; should stop early now
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=STOPPING,
            repetition_penalty=1.05,            # mild anti-looping (optional)
        )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract after "### Response:" and cut at first end tag
    response_text = raw_output.split("### Response:\n", 1)[-1]
    response_text = response_text.split("<END_OF_RESPONSE>", 1)[0]
    # Keep your regex as extra safety
    response_text = re.sub(r"<END_OF_RESPONSE.*", "", response_text, flags=re.DOTALL).strip()

    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
