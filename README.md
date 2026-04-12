# AutoStream AI Agent — Inflx Social-to-Lead Workflow

> **Machine Learning Intern Assignment | ServiceHive × Inflx**  
> A LangGraph-powered conversational AI agent that identifies high-intent social leads and captures them via a mock CRM tool.

---

## 📁 Project Structure

```
inflx-autostream/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine (core agent logic)
│   ├── intent.py         # Rule-based intent classifier
│   └── rag.py            # RAG pipeline (local JSON knowledge base)
├── knowledge_base/
│   └── autostream_kb.json  # Pricing, features, policies, FAQs
├── tools/
│   ├── __init__.py
│   └── lead_capture.py   # mock_lead_capture() tool + CRM logger
├── logs/
│   └── leads.json        # Auto-generated lead log (git-ignored)
├── main.py               # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/inflx-autostream.git
cd inflx-autostream
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
# Open .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the agent

```bash
python main.py
```

---

## 💬 Example Conversation Flow

```
You: Hi, tell me about your pricing.
Agent: AutoStream offers two plans...
       Basic Plan ($29/month): 10 videos, 720p...
       Pro Plan ($79/month): Unlimited videos, 4K, AI captions...

You: That sounds great! I want to try the Pro plan for my YouTube channel.
Agent: That's awesome! I'd love to get you set up... What's your name?

You: Alex Johnson
Agent: Nice to meet you, Alex! What's your email address?

You: alex@example.com
Agent: Got it! Which creator platform do you primarily use?

You: YouTube
[LEAD CAPTURED] → Alex Johnson | alex@example.com | YouTube
Agent: 🚀 You're all set, Alex! We've captured your details...
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over vanilla LangChain or AutoGen because it provides **explicit, inspectable state machines** — exactly what a multi-turn lead qualification workflow needs. Unlike AutoGen's agent-loop paradigm (which excels at multi-agent collaboration), LangGraph lets us define precisely *which* node runs *when* based on deterministic routing logic. This prevents unintended tool calls (like triggering `mock_lead_capture` before all three fields are collected) and makes debugging trivial.

### How State is Managed

The agent uses a single `AgentState` TypedDict that persists across every graph invocation. Key fields:

- **`messages`** — full conversation history (injected into every LLM call, enabling multi-turn memory without external storage)
- **`intent`** — classified on every turn by the rule-based `intent.py` module
- **`lead_stage`** — a finite-state machine within the state (`none → ask_name → ask_email → ask_platform → ready_to_capture → done`), ensuring fields are collected in strict sequence
- **`rag_context`** — KB snippet retrieved per turn, injected into the system prompt

This design achieves **zero-latency local memory** (no vector DB needed) while remaining production-ready — swapping to LangGraph's `SqliteSaver` or Redis checkpointer adds persistence in one line.

---

## 📱 WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, we use the **WhatsApp Business Cloud API** (Meta) with a webhook-driven FastAPI backend.

### Architecture

```
WhatsApp User
     │
     ▼
Meta Webhook POST → /webhook (FastAPI)
     │
     ▼
Message Router
     │
     ├─ Verify webhook challenge (GET)
     └─ Handle inbound message (POST)
           │
           ▼
     Load AgentState from Redis (keyed by wa_id / phone number)
           │
           ▼
     LangGraph graph.invoke(state, user_message)
           │
           ▼
     Persist updated state → Redis
           │
           ▼
     POST reply to Meta Graph API → User's WhatsApp
```

### Key Implementation Steps

**1. Set up Meta App**
- Create a Meta Developer App → Add WhatsApp product
- Get a permanent access token + Phone Number ID
- Register a webhook URL pointing to your FastAPI server

**2. FastAPI Webhook Server**

```python
# whatsapp_webhook.py
from fastapi import FastAPI, Request
import httpx, redis, json
from agent.graph import build_graph, AgentState

app = FastAPI()
r = redis.Redis()
graph = build_graph()

VERIFY_TOKEN = "inflx_secret_token"

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params["hub.challenge"])
    return {"error": "Invalid token"}, 403

@app.post("/webhook")
async def handle_message(request: Request):
    data = await request.json()
    entry = data["entry"][0]["changes"][0]["value"]
    
    if "messages" not in entry:
        return {"status": "ok"}
    
    msg = entry["messages"][0]
    wa_id = msg["from"]          # phone number = unique session key
    user_text = msg["text"]["body"]
    
    # Load or init state from Redis
    raw = r.get(f"state:{wa_id}")
    state = json.loads(raw) if raw else AgentState(
        messages=[], intent="none", lead_stage="none",
        lead_name=None, lead_email=None, lead_platform=None,
        rag_context="", response=""
    )
    
    # Add user message & invoke graph
    state["messages"].append({"role": "user", "content": user_text})
    result = graph.invoke(state)
    
    # Persist updated state
    r.set(f"state:{wa_id}", json.dumps(result), ex=86400)  # 24h TTL
    
    # Send reply via Meta API
    await send_whatsapp_message(wa_id, result["response"])
    return {"status": "ok"}

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload,
                          headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
```

**3. State Persistence**  
Each WhatsApp number (`wa_id`) maps to its own Redis key. The full `AgentState` JSON is stored with a 24-hour TTL, giving users persistent multi-turn memory across messages.

**4. Deployment**  
Host on Railway, Render, or AWS EC2. Expose HTTPS via Nginx + Let's Encrypt (Meta requires HTTPS webhooks). Use Gunicorn + Uvicorn workers for production concurrency.

---

## 🧪 Evaluation Checklist

| Criterion | Implementation |
|-----------|---------------|
| Intent Detection | Rule-based classifier in `agent/intent.py` — 3 classes |
| RAG | Keyword-scored retrieval over local JSON KB in `agent/rag.py` |
| State Management | LangGraph `AgentState` TypedDict, persists across all turns |
| Tool Calling | `mock_lead_capture()` called only after all 3 fields collected |
| Code Clarity | Modular: graph / rag / intent / tools are fully separated |
| WhatsApp Deployment | Fully documented webhook + Redis pattern above |

---

## 📄 License

MIT — Built for ServiceHive Inflx Intern Assignment.
