"""
app.py
Streamlit UI for AutoStream AI Agent - Live browser chat interface.
Run: streamlit run app.py
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, AgentState
from agent.intent import classify_intent
from agent.rag import retrieve_context
from tools.lead_capture import mock_lead_capture

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

if "state" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.agent_state: AgentState = {
        "messages": [],
        "intent": "none",
        "lead_stage": "none",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "rag_context": "",
        "response": "",
        "conversation_summary": "",
        "lead_score": 0,
    }
    st.session_state.lead_score = 0

COLORS = {
    "primary": "#FF6B35",
    "secondary": "#1A1A2E",
    "accent": "#4ECDC4",
    "background": "#0F0F1A",
    "card": "#1A1A2E",
    "text": "#FFFFFF",
    "muted": "#8B8B9A"
}

st.markdown(f"""
<style>
    .stApp {{ background: {COLORS['background']}; }}
    .chat-message {{ 
        padding: 16px 20px; 
        border-radius: 16px; 
        margin: 12px 0;
        font-family: 'Inter', sans-serif;
    }}
    .user-message {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #FF8F5A 100%);
        color: white;
        margin-left: 40px;
        box-shadow: 0 4px 20px rgba(255,107,53,0.3);
    }}
    .assistant-message {{
        background: {COLORS['card']};
        color: {COLORS['text']};
        margin-right: 40px;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .lead-score {{
        background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['primary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    .status-badge {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['accent']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        color: {COLORS['accent']};
    }}
</style>
""", unsafe_allow_html=True)

def get_intent_display(intent: str) -> str:
    mapping = {
        "greeting": "👋 Greeting",
        "product_inquiry": "🔍 Product Inquiry", 
        "high_intent": "🔥 High Intent",
        "none": "—"
    }
    return mapping.get(intent, intent)

def get_stage_display(stage: str) -> str:
    mapping = {
        "none": "New Lead",
        "ask_name": "Collecting Name",
        "ask_email": "Collecting Email",
        "ask_platform": "Collecting Platform",
        "ready_to_capture": "Ready to Capture",
        "done": "Completed"
    }
    return mapping.get(stage, stage)

st.title("🎬 AutoStream AI Assistant")
st.caption("Powered by Inflx | AI-Powered Lead Qualification")

with st.sidebar:
    st.header("📊 Session Info")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Turns", len(st.session_state.agent_state.get("messages", [])) // 2)
    with col2:
        stage = st.session_state.agent_state.get("lead_stage", "none")
        st.metric("Stage", get_stage_display(stage))
    
    st.divider()
    
    st.subheader("🎯 Lead Score")
    score = st.session_state.lead_score
    score_color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: {COLORS['card']}; border-radius: 12px;">
        <div style="font-size: 36px; font-weight: bold;">{score}</div>
        <div style="color: {COLORS['muted']};">/ 100</div>
        <div style="margin-top: 8px;">{score_color} {get_intent_display(st.session_state.agent_state.get('intent', 'none'))}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.agent_state = {
            "messages": [],
            "intent": "none", 
            "lead_stage": "none",
            "lead_name": None,
            "lead_email": None,
            "lead_platform": None,
            "rag_context": "",
            "response": "",
        }
        st.session_state.lead_score = 0
        st.rerun()

for msg in st.session_state.agent_state.get("messages", []):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f'<div class="chat-message assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Type your message...", disabled=st.session_state.agent_state.get("lead_stage") == "done"):
    state = st.session_state.agent_state
    state["messages"] = state["messages"] + [{"role": "user", "content": prompt}]
    
    result = st.session_state.graph.invoke(state)
    st.session_state.agent_state = result
    
    intent = result.get("intent", "none")
    if intent == "high_intent":
        st.session_state.lead_score = min(100, st.session_state.lead_score + 30)
    elif intent == "product_inquiry":
        st.session_state.lead_score = min(100, st.session_state.lead_score + 10)
    
    if result.get("lead_stage") == "done":
        st.session_state.lead_score = 100
    
    st.rerun()

if st.session_state.agent_state.get("lead_stage") == "done":
    st.success("🎉 Lead captured! Session complete.")