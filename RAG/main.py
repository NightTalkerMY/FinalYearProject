import uvicorn
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ProposedRouter import *
from DatabaseRouting import *
from typing import Optional

# REMOVED: from asin_finder import ASINFinder

class RAGRequest(BaseModel):
    query: str
    asin: Optional[str] = None          # Used for your gesture exit
    current_asin: Optional[str] = None  # NEW: Used for carousel focus
    is_followup: Optional[bool] = False # NEW: Tells RAG this is a carousel question

app = FastAPI()

print("--- [RAG BOOT] Initializing Models... ---")

try:
    proposed_math_router = ClusterSemanticRouter(
        anchors_path="teleoracle_v2_anchors.npz", 
        threshold=0.6
    )
    
    if 'model' not in globals():
        print("⚠️ WARNING: 'model' variable not found via imports.")
    
    wrapped_proposed_router = ProposedRouterWrapper(proposed_math_router, model)
    
    # Initialize Router
    router = DatabaseRouting(db_path="db", verbose=True, use_length_sorting=True)
    
    # REMOVED: product_lookup = ASINFinder("product.json")
    print("--- [RAG BOOT] Models Loaded Successfully. ---")

except Exception as e:
    print(f"❌ CRITICAL ERROR during RAG Startup: {e}")
    sys.exit(1)


@app.post("/get_context")
def get_context(request: RAGRequest):
    print(f"\n[RAG] 📨 Received: {request.query}", flush=True)

    try:
        # --- CASE 1: GESTURE EXIT (Lookup by ASIN) ---
        if request.query == "<GESTURE_EXIT>" and request.asin:
            print(f"[RAG] 🛑 Handling Exit for ASIN: {request.asin}", flush=True)
            context_str = router.get_product_by_asin(request.asin)
            return {
                "context": context_str, 
                "intent": "exit", 
                "trigger_carousel": False, 
                "asins": []
            }

        # --- CASE 2: CAROUSEL FOLLOW-UP QUESTION ---
        if request.is_followup and request.current_asin:
            print(f"[RAG] 🎯 Follow-up question focused on ASIN: {request.current_asin}", flush=True)
            
            # Grab the exact product the user is looking at in the carousel
            context_str = router.get_product_by_asin(request.current_asin)
            
            # We return empty ASINs so the orchestrator keeps the current carousel list intact
            return {
                "context": context_str,
                "intent": "product_followup",
                "trigger_carousel": True, 
            }

        # --- CASE 3: STANDARD SEARCH ---
        print("[RAG] 🧠 Routing...", flush=True)
        predicted_db, confidence = wrapped_proposed_router.route(request.query)
        print(f"[RAG] 🔍 Predicted: {predicted_db} ({confidence:.2f})", flush=True)
        
        search_results = router.query(request.query, predicted_db)
        
        asins_found = []
        formatted_context = ""

        if not search_results:
             formatted_context = "No products found."
             trigger_carousel = False
        
        elif predicted_db == "product":
            asins_found = [item["asin"] for item in search_results if item["asin"]]
            formatted_context = router.format_product_list(search_results)
            trigger_carousel = len(asins_found) > 0
        
        else:
            formatted_context = "\n".join([item["content"] for item in search_results])
            trigger_carousel = False

        print(f"[RAG] ✅ Done. Found {len(asins_found)} ASINs.", flush=True)
        
        return {
            "context": formatted_context,
            "intent": predicted_db,
            "trigger_carousel": trigger_carousel,
            "asins": asins_found 
        }

    except Exception as e:
        print(f"❌ [RAG ERROR]: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)