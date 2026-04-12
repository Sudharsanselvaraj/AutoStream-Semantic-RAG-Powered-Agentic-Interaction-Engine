"""
agent/intent.py
Rule-based + LLM-assisted intent classifier for AutoStream agent.
Classifies user messages into one of three intent categories.
"""

import re


# High-intent signal phrases
HIGH_INTENT_PHRASES = [
    "i want to", "i'd like to", "i would like to",
    "sign up", "subscribe", "get started", "try the pro",
    "try the basic", "purchase", "buy", "upgrade",
    "start with", "onboard", "register", "create an account",
    "join", "let's do it", "sounds good", "i'm in",
    "get it", "get the plan", "ready to", "let me try",
    "how do i sign up", "where do i sign up",
]

PRODUCT_INQUIRY_PHRASES = [
    "price", "pricing", "plan", "cost", "feature", "what does",
    "how much", "tell me about", "what is", "explain",
    "difference", "compare", "which plan", "4k", "captions",
    "resolution", "refund", "policy", "support", "unlimited",
    "basic", "pro", "cancel", "trial", "free",
]

GREETING_PHRASES = [
    "hi", "hello", "hey", "good morning", "good evening",
    "good afternoon", "what's up", "howdy", "greetings",
    "yo", "sup",
]


def classify_intent(message: str) -> str:
    """
    Classify user intent from message text.

    Returns one of:
    - "greeting"
    - "product_inquiry"
    - "high_intent"
    """
    text = message.lower().strip()
    tokens = re.sub(r"[^a-z0-9\s']", " ", text).split()

    # Check high-intent first (most specific)
    for phrase in HIGH_INTENT_PHRASES:
        if phrase in text:
            return "high_intent"

    # Check product inquiry
    for phrase in PRODUCT_INQUIRY_PHRASES:
        if phrase in text:
            return "product_inquiry"

    # Check greeting (only if very short or pure greeting)
    if len(tokens) <= 6:
        for phrase in GREETING_PHRASES:
            if phrase in tokens:
                return "greeting"

    # Default fallback
    return "product_inquiry"
