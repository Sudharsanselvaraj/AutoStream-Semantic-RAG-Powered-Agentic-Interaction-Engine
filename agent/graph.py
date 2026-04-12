"""
agent/graph.py
LangGraph-powered agent graph for AutoStream Social-to-Lead workflow.

Graph Nodes:
  - router        → classify intent and route
  - responder     → answer using RAG context
  - lead_qualify  → collect name, email, platform
  - lead_capture  → call mock_lead_capture tool
  - summarizer    → auto-summarize conversation when >8 turns

State is passed through the graph and persisted across turns.
"""

from __future__ import annotations
import os
from typing import TypedDict, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.intent import classify_intent
from agent.rag import retrieve_context
from tools.lead_capture import mock_lead_capture, calculate_lead_score


# ─────────────────────────────────────────────
# 1.  State Schema
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list                      # full conversation history
    intent: str                         # latest classified intent
    lead_stage: str                     # "none" | "ask_name" | "ask_email" | "ask_platform" | "done"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    rag_context: str                    # retrieved KB snippet
    response: str                       # agent's reply to user
    conversation_summary: str          # summarized context for long conversations
    lead_score: int                     # calculated lead score (0-100)


# ─────────────────────────────────────────────
# 2.  LLM (Grok)
# ─────────────────────────────────────────────

def _get_llm():
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError("GROK_API_KEY not set. Add it to your .env file.")
    return ChatOpenAI(
        model="grok-2",
        temperature=0.3,
        max_tokens=512,
        base_url="https://api.x.ai/v1",
        api_key=api_key
    )


def _get_summary_llm():
    """Faster/cheaper LLM for summarization."""
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError("GROK_API_KEY not set. Add it to your .env file.")
    return ChatOpenAI(
        model="grok-2",
        temperature=0.2,
        max_tokens=256,
        base_url="https://api.x.ai/v1",
        api_key=api_key
    )


# ─────────────────────────────────────────────
# 3.  Node Functions
# ─────────────────────────────────────────────

SUMMARIZE_THRESHOLD = 8  # Turn threshold for auto-summarization


def router_node(state: AgentState) -> AgentState:
    """Classify intent and fetch RAG context for the latest user message."""
    last_human = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    )
    intent = classify_intent(last_human)
    rag_ctx = retrieve_context(last_human) if intent in ("product_inquiry", "high_intent") else ""
    
    lead_score = calculate_lead_score(
        intent=intent,
        lead_stage=state.get("lead_stage", "none"),
        has_name=bool(state.get("lead_name")),
        has_email=bool(state.get("lead_email")),
        has_platform=bool(state.get("lead_platform")),
        conversation_length=len(state["messages"]) // 2
    )
    
    return {**state, "intent": intent, "rag_context": rag_ctx, "lead_score": lead_score}


def summarizer_node(state: AgentState) -> AgentState:
    """Compress conversation history into a summary when too long."""
    llm = _get_summary_llm()
    
    recent_msgs = state["messages"][-16:]  # Keep last 16 messages
    
    system = (
        "You are a conversation summarizer. Create a brief 2-3 sentence summary "
        "of the key points from this conversation so far. Focus on:\n"
        "- What the user has asked about or shown interest in\n"
        "- Any lead information collected (name, email, platform)\n"
        "- Current qualification stage\n\n"
        "Keep it concise and informative."
    )
    
    messages = [SystemMessage(content=system)]
    for m in recent_msgs:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    
    summary = llm.invoke(messages).content
    
    return {**state, "conversation_summary": summary, "messages": recent_msgs}


def responder_node(state: AgentState) -> AgentState:
    """Generate a RAG-grounded reply for greetings and product inquiries."""
    llm = _get_llm()
    
    context_parts = []
    if state.get("conversation_summary"):
        context_parts.append(f"Previous conversation summary: {state['conversation_summary']}")
    context_parts.append(f"KNOWLEDGE BASE CONTEXT:\n{state['rag_context']}")
    
    system = (
        "You are AutoStream's helpful and friendly AI assistant. "
        "Answer questions accurately using ONLY the context below. "
        "Be concise, warm, and conversational. "
        "If the user seems interested in signing up, gently encourage them.\n\n"
        + "\n\n".join(context_parts)
    )
    
    lc_messages = [SystemMessage(content=system)]
    for m in state["messages"]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    
    reply = llm.invoke(lc_messages).content
    
    updated_messages = state["messages"] + [{"role": "assistant", "content": reply}]
    return {**state, "messages": updated_messages, "response": reply}


def lead_qualify_node(state: AgentState) -> AgentState:
    """
    Collect lead details across turns (name → email → platform).
    Each call advances the stage by one field.
    """
    stage = state.get("lead_stage", "none")
    
    last_human = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    ).strip()
    
    new_state = dict(state)
    
    if stage == "none" or stage == "high_intent_detected":
        reply = (
            "That's awesome! I'd love to get you set up on the Pro plan. 🎉\n"
            "Let's get you registered. First — what's your name?"
        )
        new_state["lead_stage"] = "ask_name"
    
    elif stage == "ask_name":
        new_state["lead_name"] = last_human
        reply = f"Nice to meet you, {last_human}! What's your email address?"
        new_state["lead_stage"] = "ask_email"
    
    elif stage == "ask_email":
        new_state["lead_email"] = last_human
        reply = (
            "Got it! Last question — which creator platform do you primarily use? "
            "(e.g., YouTube, Instagram, TikTok, Twitch…)"
        )
        new_state["lead_stage"] = "ask_platform"
    
    elif stage == "ask_platform":
        new_state["lead_platform"] = last_human
        new_state["lead_stage"] = "ready_to_capture"
        reply = ""
    
    else:
        reply = "Something went wrong in qualification. Let me restart — what's your name?"
        new_state["lead_stage"] = "ask_name"
    
    if reply:
        new_state["messages"] = state["messages"] + [{"role": "assistant", "content": reply}]
        new_state["response"] = reply
    
    lead_score = calculate_lead_score(
        intent=state.get("intent", "none"),
        lead_stage=new_state.get("lead_stage", "none"),
        has_name=bool(new_state.get("lead_name")),
        has_email=bool(new_state.get("lead_email")),
        has_platform=bool(new_state.get("lead_platform")),
        conversation_length=len(state["messages"]) // 2
    )
    new_state["lead_score"] = lead_score
    
    return new_state


def lead_capture_node(state: AgentState) -> AgentState:
    """Call mock_lead_capture once all three fields are collected."""
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )
    
    reply = (
        f"🚀 You're all set, {state['lead_name']}! "
        f"We've captured your details and our team will reach out to your email **{state['lead_email']}** shortly.\n\n"
        "In the meantime, you can explore AutoStream at autostream.io. "
        "Welcome to the creator revolution! 🎬"
    )
    
    updated_messages = state["messages"] + [{"role": "assistant", "content": reply}]
    return {**state, "messages": updated_messages, "response": reply, "lead_stage": "done", "lead_score": 100}


# ─────────────────────────────────────────────
# 4.  Routing Logic
# ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["summarizer", "responder", "lead_qualify"]:
    """Decide whether to summarize, respond, or qualify."""
    stage = state.get("lead_stage", "none")
    
    if stage in ("ask_name", "ask_email", "ask_platform", "high_intent_detected"):
        return "lead_qualify"
    
    if state["intent"] == "high_intent" and stage == "none":
        return "lead_qualify"
    
    if len(state["messages"]) >= SUMMARIZE_THRESHOLD * 2:
        return "summarizer"
    
    return "responder"


def route_after_summarizer(state: AgentState) -> Literal["responder", "lead_qualify"]:
    """After summarization, decide next step."""
    stage = state.get("lead_stage", "none")
    
    if stage in ("ask_name", "ask_email", "ask_platform", "high_intent_detected"):
        return "lead_qualify"
    
    if state["intent"] == "high_intent" and stage == "none":
        return "lead_qualify"
    
    return "responder"


def route_after_qualify(state: AgentState) -> Literal["lead_capture", END]:
    if state.get("lead_stage") == "ready_to_capture":
        return "lead_capture"
    return END


# ─────────────────────────────────────────────
# 5.  Build Graph
# ─────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("router", router_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("responder", responder_node)
    builder.add_node("lead_qualify", lead_qualify_node)
    builder.add_node("lead_capture", lead_capture_node)
    
    builder.set_entry_point("router")
    
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"summarizer": "summarizer", "responder": "responder", "lead_qualify": "lead_qualify"}
    )
    
    builder.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {"responder": "responder", "lead_qualify": "lead_qualify"}
    )
    
    builder.add_edge("responder", END)
    builder.add_conditional_edges(
        "lead_qualify",
        route_after_qualify,
        {"lead_capture": "lead_capture", END: END}
    )
    builder.add_edge("lead_capture", END)
    
    return builder.compile()