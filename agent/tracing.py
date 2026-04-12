"""
agent/tracing.py
LangSmith observability integration for agent tracing.
Add 5 env vars to .env to enable full trace dashboards.
"""

import os
from typing import Optional


def get_tracing_config() -> Optional[dict]:
    """
    Get LangSmith tracing configuration from environment.
    
    Add these to your .env:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls-...
    LANGCHAIN_PROJECT=autostream-agent
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    """
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        return None
    
    return {
        "project_name": os.getenv("LANGCHAIN_PROJECT", "autostream-default"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "api_key": os.getenv("LANGCHAIN_API_KEY", ""),
    }


def enable_tracing():
    """Enable LangSmith tracing via environment variables."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    required_vars = ["LANGCHAIN_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"⚠️ LangSmith: Missing env vars: {missing}")
        return False
    
    print(f"✅ LangSmith tracing enabled: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    return True


def create_traced_llm(llm, project_name: str = "autostream-agent"):
    """
    Wrap an LLM with LangSmith tracing.
    
    Usage:
        from langchain_anthropic import ChatAnthropic
        from agent.tracing import create_traced_llm
        
        base_llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
        traced_llm = create_traced_llm(base_llm)
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tracers import LangChainTracer
        from langchain_core.tracers.context import tracing_context
        
        tracer = LangChainTracer(
            project_name=project_name,
            example_id=None
        )
        
        return llm.with_config({"callbacks": [tracer]})
    except ImportError:
        print("⚠️ LangSmith dependencies not installed. Run: pip install langchain[langsmith]")
        return llm


def trace_event(event_name: str, metadata: dict = None):
    """Log an event to LangSmith if tracing is enabled."""
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        return
    
    try:
        from langsmith import traceable
        
        @traceable(name=event_name)
        def log_event():
            return metadata or {}
        
        log_event()
    except Exception as e:
        print(f"⚠️ Failed to trace event {event_name}: {e}")


if __name__ == "__main__":
    print("LangSmith Tracing Configuration")
    print("=" * 40)
    config = get_tracing_config()
    if config:
        print(f"Project: {config['project_name']}")
        print(f"Endpoint: {config['endpoint']}")
    else:
        print("Not configured. Add to .env:")
        print("  LANGCHAIN_TRACING_V2=true")
        print("  LANGCHAIN_API_KEY=ls-...")
        print("  LANGCHAIN_PROJECT=autostream-agent")