"""
main.py
CLI entrypoint for the AutoStream Social-to-Lead Agent (Inflx).

Run:
    python main.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, AgentState

BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Assistant  (powered by Inflx)  ║
║   Type 'quit' or 'exit' to end the conversation.    ║
╚══════════════════════════════════════════════════════╝
"""

INITIAL_STATE: AgentState = {
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


def run():
    print(BANNER)

    graph = build_graph()
    state = INITIAL_STATE.copy()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Thanks for chatting! Have a great day. 👋")
            break

        # Append user message to state
        state["messages"] = state["messages"] + [{"role": "user", "content": user_input}]

        # Run graph
        result = graph.invoke(state)

        # Update persistent state
        state = result

        # Print agent response
        print(f"\nAgent: {state['response']}\n")

        # If lead captured, optionally end
        if state.get("lead_stage") == "done":
            print("─" * 55)
            print("✅  Lead capture complete. Session ended.")
            break


if __name__ == "__main__":
    run()
