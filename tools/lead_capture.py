"""
tools/lead_capture.py
Enhanced lead capture tool with email validation and duplicate detection.
"""

import json
import os
import re
from datetime import datetime
from typing import Optional


EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

LOG_PATH = os.path.join(os.path.dirname(__file__), "../logs/leads.json")


def _load_leads() -> list:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return []


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def is_duplicate(email: str) -> bool:
    """Check if email already exists in leads.json."""
    leads = _load_leads()
    return any(lead.get("email", "").lower() == email.strip().lower() for lead in leads)


def calculate_lead_score(
    intent: str,
    lead_stage: str,
    has_name: bool,
    has_email: bool,
    has_platform: bool,
    conversation_length: int
) -> int:
    """
    Calculate lead score (0-100) based on multiple factors.
    
    Factors:
    - Intent type (high intent = more points)
    - Lead stage progress
    - Data completeness
    - Engagement level
    """
    score = 0
    
    if intent == "high_intent":
        score += 40
    elif intent == "product_inquiry":
        score += 20
    elif intent == "greeting":
        score += 5
    
    stage_scores = {
        "none": 0,
        "high_intent_detected": 5,
        "ask_name": 10,
        "ask_email": 20,
        "ask_platform": 30,
        "ready_to_capture": 40,
        "done": 50
    }
    score += stage_scores.get(lead_stage, 0)
    
    if has_name:
        score += 10
    if has_email:
        score += 15
    if has_platform:
        score += 10
    
    min_engagement = min(conversation_length * 2, 15)
    score += min_engagement
    
    return min(100, score)


def mock_lead_capture(
    name: str,
    email: str,
    platform: str,
    validate: bool = True
) -> dict:
    """
    Enhanced mock API function to capture a qualified lead.
    
    Args:
        name: Lead's full name
        email: Lead's email address
        platform: Creator platform (YouTube, Instagram, etc.)
        validate: Whether to validate email and check duplicates
    
    Returns:
        dict with success status, message, and optional warnings
    """
    warnings = []
    
    if validate:
        if not validate_email(email):
            return {
                "success": False,
                "error": "Invalid email format",
                "message": "Please provide a valid email address."
            }
        
        if is_duplicate(email):
            return {
                "success": False,
                "error": "Duplicate lead",
                "message": "This email is already registered."
            }
    
    print("\n" + "=" * 50)
    print("🎯  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")
    
    lead = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat(),
        "source": "Inflx-AutoStream-Agent"
    }
    
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    leads = _load_leads()
    leads.append(lead)
    with open(LOG_PATH, "w") as f:
        json.dump(leads, f, indent=2)
    
    return {
        "success": True,
        "message": f"Lead for {name} captured and logged to CRM.",
        "lead_id": f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "warnings": warnings if warnings else None
    }