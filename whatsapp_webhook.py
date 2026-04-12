"""
whatsapp_webhook.py
FastAPI server for WhatsApp Business Cloud API webhook integration.
Run: uvicorn whatsapp_webhook:app --reload --port 8000
"""

import os
import json
import httpx
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, AgentState
from tools.lead_capture import calculate_lead_score

app = FastAPI(title="AutoStream WhatsApp Webhook")

GRAPH = build_graph()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "inflx_secret_token")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")

LEADS_FILE = "logs/leads.json"


def load_state(wa_id: str) -> dict:
    """Load agent state from local JSON file (simulating Redis)."""
    state_file = f"logs/state_{wa_id}.json"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def save_state(wa_id: str, state: dict):
    """Save agent state to local JSON file (simulating Redis)."""
    os.makedirs("logs", exist_ok=True)
    state_file = f"logs/state_{wa_id}.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def create_initial_state() -> AgentState:
    """Create fresh agent state."""
    return {
        "messages": [],
        "intent": "none",
        "lead_stage": "none",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "rag_context": "",
        "response": "",
        "conversation_summary": "",
        "lead_score": 0
    }


async def send_whatsapp_message(to: str, text: str) -> bool:
    """Send reply via Meta Graph API."""
    if not PHONE_NUMBER_ID or not ACCESS_TOKEN:
        print("⚠️ WhatsApp credentials not configured")
        return False
    
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            return response.status_code == 200
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")
        return False


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    WhatsApp webhook verification endpoint.
    Meta sends a GET request to verify the webhook.
    """
    params = dict(request.query_params)
    
    if params.get("hub.mode") == "subscribe":
        if params.get("hub.verify_token") == VERIFY_TOKEN:
            return int(params["hub.challenge"])
        else:
            raise HTTPException(status_code=403, detail="Invalid verify token")
    
    return {"status": "ok"}


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle incoming WhatsApp messages.
    Meta sends POST requests when users message the business account.
    """
    data = await request.json()
    
    if "entry" not in data or not data["entry"]:
        return {"status": "ok"}
    
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        
        if "messages" not in entry:
            return {"status": "ok"}
        
        messages = entry["messages"]
        if not messages:
            return {"status": "ok"}
        
        msg = messages[0]
        wa_id = msg["from"]
        user_text = msg.get("text", {}).get("body", "").strip()
        
        if not user_text:
            return {"status": "ok"}
        
        print(f"\n📱 WhatsApp message from {wa_id}: {user_text}")
        
        existing_state = load_state(wa_id)
        if existing_state:
            state = existing_state
        else:
            state = create_initial_state()
        
        state["messages"] = state["messages"] + [{"role": "user", "content": user_text}]
        
        result = GRAPH.invoke(state)
        
        save_state(wa_id, result)
        
        if result.get("response"):
            sent = await send_whatsapp_message(wa_id, result["response"])
            if sent:
                print(f"✅ Reply sent to {wa_id}")
            else:
                print(f"❌ Failed to send reply to {wa_id}")
        
        return {"status": "ok"}
    
    except Exception as e:
        print(f"Error handling webhook: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "AutoStream WhatsApp Webhook",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/"
        }
    }


@app.get("/state/{wa_id}")
async def get_state(wa_id: str):
    """Debug endpoint to check conversation state."""
    state = load_state(wa_id)
    if state:
        return {"wa_id": wa_id, "state": state}
    return {"error": "No state found for this wa_id"}


@app.post("/reset/{wa_id}")
async def reset_state(wa_id: str):
    """Reset conversation state for a specific WhatsApp user."""
    state_file = f"logs/state_{wa_id}.json"
    if os.path.exists(state_file):
        os.remove(state_file)
    return {"status": "reset", "wa_id": wa_id}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AutoStream WhatsApp Webhook Server...")
    print(f"   Webhook URL: /webhook")
    print(f"   Health check: /")
    uvicorn.run(app, host="0.0.0.0", port=8000)