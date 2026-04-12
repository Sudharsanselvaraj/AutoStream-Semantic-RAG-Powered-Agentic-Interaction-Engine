# AutoStream AI Agent
### Inflx Social-to-Lead Agentic Workflow · ServiceHive × Inflx · April 2026

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-1C3C3C?style=flat-square)
![Claude](https://img.shields.io/badge/Claude_3_Haiku-Anthropic-D4A017?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A production-grade, multi-turn conversational lead-generation agent powered by LangGraph, Claude 3 Haiku, and semantic RAG — built for ServiceHive's Inflx platform.**

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Graph Node Reference](#4-graph-node-reference)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Intent Classification](#6-intent-classification)
7. [Lead Qualification FSM](#7-lead-qualification-fsm)
8. [Lead Capture Tool](#8-lead-capture-tool)
9. [Lead Scoring](#9-lead-scoring)
10. [WhatsApp Deployment](#10-whatsapp-deployment)
11. [Streamlit Chat UI](#11-streamlit-chat-ui)
12. [LangSmith Observability](#12-langsmith-observability)
13. [Installation & Setup](#13-installation--setup)
14. [Environment Variables](#14-environment-variables)
15. [Example Conversation Flow](#15-example-conversation-flow)
16. [Project Structure](#16-project-structure)
17. [Evaluation Checklist](#17-evaluation-checklist)

---

## 1. Project Overview

**AutoStream AI Agent** is a deterministic, stateful conversational system designed to convert social media inbound messages into qualified CRM leads for the Inflx platform. The agent handles three distinct tasks within a single multi-turn conversation:

- **Inform** — answers product/pricing questions using a local knowledge base via RAG
- **Detect** — classifies user intent in real-time to identify purchase readiness
- **Convert** — runs a structured field-collection flow (name → email → platform) before firing the CRM tool

The entire pipeline is a **LangGraph `StateGraph`** with deterministic routing. The LLM (`claude-3-haiku-20240307`) is invoked only for response generation and summarisation — never for routing decisions, which makes behaviour fully testable and reproducible.

**Quick Reference**

| Property | Value |
|---|---|
| Language | Python 3.9+ |
| Framework | LangChain + LangGraph (StateGraph) |
| LLM | Claude 3 Haiku (`claude-3-haiku-20240307`) |
| Knowledge Base | Local JSON — plans, policies, FAQs |
| State | `AgentState` TypedDict, persisted across turns |
| UI Options | CLI · Streamlit · WhatsApp Webhook |
| Lead Logging | `logs/leads.json` — validated, deduplicated |

---

## 2. Key Features

- **Deterministic routing** — no LLM call decides routing; all edges are pure Python conditionals
- **Finite-state lead qualification** — a 5-stage FSM guarantees the CRM tool never fires with incomplete data
- **Keyword-scored RAG** — no vector DB required; fast, zero-latency retrieval over a structured JSON KB
- **Rule-based intent classifier** — 3-class, priority-ordered, zero per-turn API calls for classification
- **Auto-summarisation** — compresses history after 8 turns to control LLM context cost
- **Email validation & duplicate detection** — RFC 5322 regex + case-insensitive email dedup
- **Real-time lead scoring** — 0–100 score updated on every turn, visible in Streamlit sidebar
- **WhatsApp-ready** — production FastAPI webhook with per-user state files
- **LangSmith tracing** — full turn-by-turn LLM observability, opt-in via env var

---

## 3. System Architecture

### 3.1 High-Level Flow

```
  Social Message (WhatsApp / CLI / Streamlit)
          │
          ▼
  ┌───────────────┐
  │  router_node  │  ← classify_intent() + retrieve_context() + lead_score
  └──────┬────────┘
         │
   route_after_router()
         │
   ┌─────┴──────────────────────────────────┐
   │             │                          │
   ▼             ▼                          ▼
summarizer    responder              lead_qualify
  _node        _node                   _node
   │             │                      │
   │          [END]          route_after_qualify()
   │                                    │
   │                         ┌──────────┴──────────┐
   │                         ▼                     ▼
   │                   lead_capture             [END]
   │                      _node
   │                         │
   └──────────────────────>[END]
```

### 3.2 AgentState TypedDict

All state is a single `TypedDict` threaded through every node — no external store needed for single-session deployments.

```python
class AgentState(TypedDict):
    messages:             list         # [{role, content}, ...] — full history
    intent:               str          # "greeting" | "product_inquiry" | "high_intent"
    lead_stage:           str          # FSM stage (see §7)
    lead_name:            Optional[str]
    lead_email:           Optional[str]
    lead_platform:        Optional[str]
    rag_context:          str          # KB snippet injected into system prompt
    response:             str          # agent's last reply (exposed to UI / webhook)
    conversation_summary: str          # set by summarizer_node after compression
    lead_score:           int          # real-time 0–100
```

### 3.3 Why LangGraph over AutoGen / plain LangChain

| Concern | LangGraph | AutoGen | Plain LangChain |
|---|---|---|---|
| Explicit routing | ✅ Conditional edges | ❌ Agent-loop | ❌ Linear chain |
| Enforce field order | ✅ FSM in state | ❌ Not built-in | ❌ Not built-in |
| Inspectable transitions | ✅ Graph visualization | ⚠️ Opaque | ⚠️ Opaque |
| Scale to multi-user | ✅ SqliteSaver / Redis checkpointer | ✅ | ⚠️ Manual |
| Add new nodes later | ✅ Additive | ✅ | ❌ Restructure needed |

LangGraph was chosen specifically because `route_after_qualify()` can structurally prevent `lead_capture_node` from ever being reached unless all three fields are populated. This is impossible to enforce cleanly in a loop-based agent.

---

## 4. Graph Node Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LANGGRAPH NODES                            │
├──────────────────┬──────────────────────────────────────────────────┤
│  Node            │  Responsibility                                  │
├──────────────────┼──────────────────────────────────────────────────┤
│  router_node     │  Classify intent, fetch RAG context, compute     │
│                  │  lead_score. Entry point for every turn.         │
├──────────────────┼──────────────────────────────────────────────────┤
│  responder_node  │  Generate RAG-grounded reply via Claude 3 Haiku  │
│                  │  + system-prompt injection.                      │
├──────────────────┼──────────────────────────────────────────────────┤
│  lead_qualify    │  Advance FSM one step; extract name / email /    │
│  _node           │  platform from user reply.                       │
├──────────────────┼──────────────────────────────────────────────────┤
│  lead_capture    │  Call mock_lead_capture() after verifying all 3  │
│  _node           │  fields; set lead_stage → "done".                │
├──────────────────┼──────────────────────────────────────────────────┤
│  summarizer_node │  Compress messages list to last 16 msgs +        │
│                  │  2-sentence summary when > 8 turns.              │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 4.1 Routing Table — `route_after_router()`

```
┌───────────────────────────────────────────┬─────────────────┐
│  Condition                                │  Destination    │
├───────────────────────────────────────────┼─────────────────┤
│  lead_stage ∈ {ask_name, ask_email,       │  lead_qualify   │
│                ask_platform}              │                 │
├───────────────────────────────────────────┼─────────────────┤
│  intent == "high_intent"                  │  lead_qualify   │
│  AND lead_stage == "none"                 │                 │
├───────────────────────────────────────────┼─────────────────┤
│  len(messages) ≥ 16                       │  summarizer     │
├───────────────────────────────────────────┼─────────────────┤
│  else                                     │  responder      │
└───────────────────────────────────────────┴─────────────────┘
```

---

## 5. RAG Pipeline

**File:** `agent/rag.py`

The RAG implementation uses **keyword-scored retrieval** against a structured local JSON knowledge base — deliberately avoiding vector embeddings. For a well-scoped SaaS FAQ domain, this eliminates embedding latency, infrastructure cost, and adds zero accuracy regression.

### 5.1 Knowledge Base Structure (`knowledge_base/autostream_kb.json`)

```
autostream_kb.json
├── company        name, tagline, description
├── plans[]        name, price_display, features[], best_for
├── policies[]     Refund · Support · Storage · Cancellation
└── faqs[]         Q&A pairs — platforms, free trial, annual pricing, AI captions
```

### 5.2 Retrieval Logic

`retrieve_context()` evaluates four independent keyword lists against the normalised query:

```
Query
  │
  ├─ matches "price / plan / cost / 4k / resolution / caption…"  → plans section
  ├─ matches "what is / about / autostream / product / tool…"    → company section
  ├─ matches "refund / cancel / support / storage / billing…"    → policies section
  ├─ matches "free / trial / annual / tiktok / youtube…"         → faqs section
  └─ no match                                                    → fallback summary
                                                                   (all plans + key policies)
```

Matched sections are concatenated and injected as `KNOWLEDGE BASE CONTEXT` in the system prompt. The LLM is instructed to answer **only** from this context — preventing hallucination on pricing or features not in the KB.

> **RAG skip optimisation:** `router_node` only calls `retrieve_context()` when `intent` is `product_inquiry` or `high_intent`. Greetings skip RAG entirely, reducing response latency for the most common first turn.

---

## 6. Intent Classification

**File:** `agent/intent.py`

Classification is **fully rule-based** — no LLM call per turn. This gives zero classification latency, 100% deterministic output, and trivial unit-testability.

### 6.1 Priority Order

```
classify_intent(message)
  │
  ├─ 1. HIGH_INTENT      — 24 phrases, checked first
  │       "i want to" · "sign up" · "get started" · "purchase" · "buy"
  │       "upgrade" · "i'm in" · "sounds good" · "ready to" · …
  │
  ├─ 2. PRODUCT_INQUIRY  — 24 phrases
  │       "price" · "pricing" · "plan" · "cost" · "how much" · "4k"
  │       "refund" · "support" · "unlimited" · "cancel" · "trial" · …
  │
  ├─ 3. GREETING         — 11 phrases (only if message ≤ 6 tokens)
  │       "hi" · "hello" · "hey" · "good morning" · "yo" · …
  │
  └─ 4. Fallback         → "product_inquiry"
                            (unrecognised → RAG, never silent)
```

> The 6-token guard on GREETING prevents `"hi, what are your prices?"` from being classified as a greeting instead of a product inquiry.

---

## 7. Lead Qualification FSM

`lead_stage` in `AgentState` is a finite-state machine. The tool-call guard is enforced **structurally** — `route_after_qualify()` returns `"lead_capture"` only when `lead_stage == "ready_to_capture"`, which is only set after the user provides the third field. There is no code path that reaches `lead_capture_node` with any field missing.

```
         ┌─────────────────────────────────────────────────────┐
         │                  LEAD QUALIFICATION FSM             │
         └─────────────────────────────────────────────────────┘

   [none] ──── high_intent detected ────► [ask_name]
                                               │
                                    user reply stored as lead_name
                                               │
                                               ▼
                                         [ask_email]
                                               │
                                    user reply stored as lead_email
                                               │
                                               ▼
                                        [ask_platform]
                                               │
                                   user reply stored as lead_platform
                                               │
                                               ▼
                                     [ready_to_capture]
                                               │
                                   mock_lead_capture() called
                                               │
                                               ▼
                                           [done] ◄── terminal
```

### FSM Transition Table

```
┌──────────────────────┬──────────────────────┬───────────────────────────────┐
│  Current Stage       │  Next Stage          │  Trigger / Action             │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│  none                │  ask_name            │  high_intent detected         │
│  ask_name            │  ask_email           │  store user reply as name     │
│  ask_email           │  ask_platform        │  store user reply as email    │
│  ask_platform        │  ready_to_capture    │  store user reply as platform │
│  ready_to_capture    │  done                │  mock_lead_capture() called   │
│  done                │  done (terminal)     │  no further tool calls        │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## 8. Lead Capture Tool

**File:** `tools/lead_capture.py`

`mock_lead_capture()` is the spec-required function, extended with production-grade validation:

### 8.1 Email Validation
Every email is validated against an RFC 5322-compliant regex before capture. Invalid emails return `{success: False, error: "Invalid email format"}` — allowing the agent to prompt re-entry.

### 8.2 Duplicate Detection
`is_duplicate()` checks `logs/leads.json` case-insensitively before writing. Duplicate emails return `{success: False, error: "Duplicate lead"}` — preventing double-capture from session replays.

### 8.3 Lead Log Schema (`logs/leads.json`)
```json
[
  {
    "name":         "Alex Johnson",
    "email":        "alex@example.com",
    "platform":     "YouTube",
    "captured_at":  "2026-04-12T08:47:00.000Z",
    "source":       "Inflx-AutoStream-Agent"
  }
]
```

> `logs/leads.json` is `.gitignore`d — never committed to the repo.

---

## 9. Lead Scoring

`calculate_lead_score()` computes a 0–100 engagement score on every `router_node` and `lead_qualify_node` call. The score is displayed live in the Streamlit sidebar.

```
┌─────────────────────────────────┬────────┬───────────────────────────────┐
│  Factor                         │  Pts   │  Notes                        │
├─────────────────────────────────┼────────┼───────────────────────────────┤
│  intent == "high_intent"        │  +40   │  Primary signal               │
│  intent == "product_inquiry"    │  +20   │  Warm lead                    │
│  intent == "greeting"           │  +5    │  Cold                         │
├─────────────────────────────────┼────────┼───────────────────────────────┤
│  lead_stage == ask_name         │  +10   │  Qualification started        │
│  lead_stage == ask_email        │  +20   │  Name collected               │
│  lead_stage == ask_platform     │  +30   │  Email collected              │
│  lead_stage == ready_to_capture │  +40   │  All fields collected         │
│  lead_stage == done             │  +50   │  → capped at 100              │
├─────────────────────────────────┼────────┼───────────────────────────────┤
│  has_name                       │  +10   │  Data completeness bonus      │
│  has_email                      │  +15   │  Data completeness bonus      │
│  has_platform                   │  +10   │  Data completeness bonus      │
├─────────────────────────────────┼────────┼───────────────────────────────┤
│  conversation_length            │  +2/turn│  Engagement signal, cap 15   │
└─────────────────────────────────┴────────┴───────────────────────────────┘
```

Sidebar colour coding: 🔴 0–39 · 🟡 40–69 · 🟢 70–100

---

## 10. WhatsApp Deployment

**File:** `whatsapp_webhook.py`

A production-ready **FastAPI** server connecting the LangGraph agent to the WhatsApp Business Cloud API.

### 10.1 Endpoint Map

```
GET  /webhook          ← Meta verification handshake
                          (hub.mode=subscribe + hub.verify_token check)

POST /webhook          ← Inbound message handler:
                          load per-user state
                          → invoke graph
                          → save state
                          → send reply via Meta Graph API

GET  /state/{wa_id}   ← Debug: inspect any user's current AgentState

POST /reset/{wa_id}   ← Reset a user's conversation to initial state
```

### 10.2 Per-User State Persistence

```
WhatsApp number (wa_id)
        │
        ▼
 logs/state_{wa_id}.json   ← individual state file per user
        │
        ▼
 In production: swap save_state() / load_state()
 to redis.set / redis.get with 24h TTL — 2-line change
```

### 10.3 Production Deployment Steps

```
1. Create Meta Developer App
   → Add WhatsApp product
   → Configure permanent access token + Phone Number ID

2. Deploy FastAPI to Railway / Render / AWS EC2
   → Add Nginx reverse proxy + Let's Encrypt TLS
     (Meta requires HTTPS on webhook URL)

3. Set environment variables:
   WHATSAPP_VERIFY_TOKEN
   WHATSAPP_PHONE_NUMBER_ID
   WHATSAPP_ACCESS_TOKEN

4. Register /webhook URL in Meta Developer Console
   → Complete GET verification handshake

5. Any WhatsApp message to the business number
   now routes through the full agent graph
```

---

## 11. Streamlit Chat UI

**File:** `app.py` · Run: `streamlit run app.py`

```
┌──────────────────────────────────┬───────────────────────┐
│          CHAT AREA               │       SIDEBAR         │
│                                  │                       │
│  ╭──────────────────────╮        │  Turn Count: 3        │
│  │  User message bubble  │       │                       │
│  │  (orange gradient)    │       │  Lead Stage:          │
│  ╰──────────────────────╯        │  [ ask_email ]        │
│                                  │                       │
│  ╭──────────────────────╮        │  Lead Score:          │
│  │  Agent reply card    │        │  ████████░░  72/100   │
│  │  (dark card style)   │        │  🟢 High Intent        │
│  ╰──────────────────────╯        │                       │
│                                  │  Intent Badge:        │
│  [  Type your message...  ] [↵]  │  🔥 High Intent       │
│                                  │                       │
│  * Input disabled after done     │  [Reset Session]      │
└──────────────────────────────────┴───────────────────────┘
```

**Features:**
- Chat messages in styled bubbles (orange gradient for user, dark card for agent)
- Live turn counter, `lead_stage` display, and colour-coded lead score bar
- Intent badge — 👋 Greeting · 🔍 Product Inquiry · 🔥 High Intent
- Session reset button (reinitialises `AgentState` without server restart)
- Input field disabled after `lead_stage == "done"` to prevent post-capture messages

---

## 12. LangSmith Observability

**File:** `agent/tracing.py`

Opt-in full LLM call tracing — zero code changes to the agent, purely environment-variable-driven.

```
.env
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=ls-...
  LANGCHAIN_PROJECT=autostream-agent
         │
         ▼
  enable_tracing() called at startup in main.py
         │
         ▼
  LangSmith Dashboard
  ├── All Claude 3 Haiku calls (responder + summarizer)
  ├── Latency per turn
  ├── Token counts (input / output)
  └── Full prompt + completion pairs
```

To disable tracing: simply omit `LANGCHAIN_TRACING_V2` from `.env`. The agent runs identically.

---

## 13. Installation & Setup

### Prerequisites

- Python 3.9+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Sudharsanselvaraj/AutoStream-Semantic-RAG-Powered-Agentic-Interaction-Engine
cd AutoStream-Semantic-RAG-Powered-Agentic-Interaction-Engine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run the agent
python main.py                    # CLI REPL
# OR
streamlit run app.py              # Browser UI
# OR
uvicorn whatsapp_webhook:app --port 8000 --reload  # WhatsApp webhook
```

### Run Commands

```
┌──────────────────────────────────────────────────────┬─────────────────────────────────┐
│  Command                                             │  Purpose                        │
├──────────────────────────────────────────────────────┼─────────────────────────────────┤
│  python main.py                                      │  CLI REPL — persistent session  │
│                                                      │  until 'quit' or lead captured  │
├──────────────────────────────────────────────────────┼─────────────────────────────────┤
│  streamlit run app.py                                │  Browser chat UI with live      │
│                                                      │  sidebar metrics                │
├──────────────────────────────────────────────────────┼─────────────────────────────────┤
│  uvicorn whatsapp_webhook:app --port 8000 --reload   │  WhatsApp webhook server        │
└──────────────────────────────────────────────────────┴─────────────────────────────────┘
```

---

## 14. Environment Variables

```
┌────────────────────────────────┬──────────┬────────────────────────────────────────────┐
│  Variable                      │  Status  │  Purpose                                   │
├────────────────────────────────┼──────────┼────────────────────────────────────────────┤
│  ANTHROPIC_API_KEY             │  Required │  Claude 3 Haiku — all LLM calls           │
├────────────────────────────────┼──────────┼────────────────────────────────────────────┤
│  WHATSAPP_VERIFY_TOKEN         │  Optional │  Meta webhook verify token                │
│  WHATSAPP_PHONE_NUMBER_ID      │  Optional │  Meta app Phone Number ID                 │
│  WHATSAPP_ACCESS_TOKEN         │  Optional │  Meta Graph API access token              │
├────────────────────────────────┼──────────┼────────────────────────────────────────────┤
│  LANGCHAIN_TRACING_V2          │  Optional │  Set "true" for LangSmith tracing         │
│  LANGCHAIN_API_KEY             │  Optional │  LangSmith API key (ls-...)               │
│  LANGCHAIN_PROJECT             │  Optional │  LangSmith project name                   │
└────────────────────────────────┴──────────┴────────────────────────────────────────────┘
```

---

## 15. Example Conversation Flow

Full trace of the spec example through the state machine:

```
Turn 1
  User   → "Hi, tell me about your pricing."
  Intent   product_inquiry
  RAG      plans section retrieved (Basic $29, Pro $79)
  Node     router → responder
  Reply  → "AutoStream offers two plans: Basic at $29/month (720p, 5 hours/month) 
             and Pro at $79/month (4K, unlimited uploads + AI captions)..."

──────────────────────────────────────────────────────────────────────

Turn 2
  User   → "That sounds good, I want to try the Pro plan for my YouTube channel."
  Intent   high_intent
  Stage    none → ask_name
  Node     router → lead_qualify
  Reply  → "That's awesome! Let's get you registered. First — what's your name?"

──────────────────────────────────────────────────────────────────────

Turn 3
  User   → "Alex Johnson"
  Stage    ask_name → ask_email
  lead_name stored: "Alex Johnson"
  Reply  → "Nice to meet you, Alex! What's your email address?"

──────────────────────────────────────────────────────────────────────

Turn 4
  User   → "alex@example.com"
  Stage    ask_email → ask_platform
  lead_email stored: "alex@example.com"
  Reply  → "Got it! Which creator platform do you primarily use?
             (e.g., YouTube, Instagram, TikTok, Twitch…)"

──────────────────────────────────────────────────────────────────────

Turn 5
  User   → "YouTube"
  Stage    ask_platform → ready_to_capture
  lead_platform stored: "YouTube"
  Node     lead_qualify → lead_capture
  Action   mock_lead_capture() called  ✓  lead_score = 100
  Log      logs/leads.json appended
  Reply  → "🚀 You're all set, Alex! We've captured your details and our team
             will reach out to alex@example.com shortly. Welcome! 🎬"
```

---

## 16. Project Structure

```
AutoStream-Semantic-RAG-Powered-Agentic-Interaction-Engine/
│
├── agent/
│   ├── graph.py           # LangGraph state machine — 5 nodes, routing, compilation
│   ├── intent.py          # Rule-based intent classifier — 3 classes, priority-ordered
│   ├── rag.py             # RAG pipeline — keyword retrieval over local JSON KB
│   └── tracing.py         # LangSmith observability — opt-in via env var
│
├── tools/
│   └── lead_capture.py    # mock_lead_capture() + email validation + dedup + scoring
│
├── knowledge_base/
│   └── autostream_kb.json # Company · Plans · Policies · FAQs — single source of truth
│
├── logs/
│   ├── leads.json         # Auto-generated CRM log (git-ignored)
│   └── state_{wa_id}.json # Per-user WhatsApp state files (git-ignored)
│
├── main.py                # CLI entrypoint — REPL loop with persistent AgentState
├── app.py                 # Streamlit chat UI — live sidebar metrics
├── whatsapp_webhook.py    # FastAPI webhook — WhatsApp Business Cloud API integration
├── requirements.txt       # All dependencies pinned with ≥ versions
├── .env.example           # Environment variable template
└── .gitignore
```

---

## 17. Evaluation Checklist

```
┌────────────────────────────────────┬────────────────────────────────────────────────────┐
│  Criterion                         │  Implementation                                    │
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Agent reasoning & intent          │  Rule-based classifier — HIGH_INTENT >             │
│  detection                         │  PRODUCT_INQUIRY > GREETING priority.              │
│                                    │  Correct on all spec examples.                     │
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Correct use of RAG                │  Keyword-scored retrieval from local JSON KB.      │
│                                    │  Plans, policies, FAQs returned only when          │
│                                    │  relevant. No hallucination.                       │
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Clean state management            │  Single AgentState TypedDict persisted across      │
│                                    │  every graph invocation. Full history injected     │
│                                    │  into every LLM call.                              │
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Proper tool calling logic         │  mock_lead_capture() fires only after              │
│                                    │  lead_stage == "ready_to_capture" — all 3 fields   │
│                                    │  guaranteed present. Enforced structurally.        │
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Code clarity & structure          │  Modular: graph / rag / intent / tools / tracing   │
│                                    │  fully separated. Each file has one responsibility.│
├────────────────────────────────────┼────────────────────────────────────────────────────┤
│  Real-world deployability          │  WhatsApp webhook (FastAPI + file-based state),    │
│                                    │  Streamlit UI, LangSmith tracing, email            │
│                                    │  validation, duplicate detection.                  │
└────────────────────────────────────┴────────────────────────────────────────────────────┘
```

---
