"""
agent/rag.py
RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Uses simple keyword + semantic scoring over a local JSON knowledge base.
"""

import json
import os
import re
from typing import Optional


KB_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/autostream_kb.json")


def _load_kb() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def retrieve_context(query: str) -> str:
    """
    Given a user query, retrieve the most relevant sections from the knowledge base.
    Returns a formatted string to be injected into the LLM prompt as context.
    """
    kb = _load_kb()
    query_lower = _normalize(query)

    sections = []

    # --- Company info ---
    if any(w in query_lower for w in ["what is", "about", "autostream", "product", "tool", "platform"]):
        c = kb["company"]
        sections.append(f"Company: {c['name']}\nTagline: {c['tagline']}\nDescription: {c['description']}")

    # --- Pricing / Plans ---
    pricing_keywords = ["price", "pricing", "plan", "cost", "basic", "pro", "month",
                        "4k", "720p", "resolution", "video", "unlimited", "caption"]
    if any(w in query_lower for w in pricing_keywords):
        plan_texts = []
        for plan in kb["plans"]:
            features_str = "\n  - ".join(plan["features"])
            plan_texts.append(
                f"Plan: {plan['name']}\n"
                f"Price: {plan['price_display']}\n"
                f"Best For: {plan['best_for']}\n"
                f"Features:\n  - {features_str}"
            )
        sections.append("=== AutoStream Plans ===\n" + "\n\n".join(plan_texts))

    # --- Policies ---
    policy_keywords = ["refund", "cancel", "support", "policy", "storage", "billing", "24/7", "days"]
    if any(w in query_lower for w in policy_keywords):
        policy_texts = [f"{p['topic']}: {p['detail']}" for p in kb["policies"]]
        sections.append("=== Policies ===\n" + "\n".join(policy_texts))

    # --- FAQs ---
    faq_keywords = ["free", "trial", "try", "annual", "yearly", "upload", "tiktok",
                    "instagram", "youtube", "language", "accuracy"]
    if any(w in query_lower for w in faq_keywords):
        faq_texts = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in kb["faqs"]]
        sections.append("=== FAQs ===\n" + "\n\n".join(faq_texts))

    # Fallback: return full summary
    if not sections:
        c = kb["company"]
        plans_summary = "; ".join(
            f"{p['name']} ({p['price_display']})" for p in kb["plans"]
        )
        sections.append(
            f"Company: {c['name']} — {c['tagline']}\n"
            f"Plans Available: {plans_summary}\n"
            f"Policies: Refunds within 7 days only. 24/7 support on Pro plan."
        )

    return "\n\n".join(sections)
