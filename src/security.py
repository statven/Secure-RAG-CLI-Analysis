# src/security.py
import re
from typing import Dict, Any

SENSITIVE_KEYWORDS = [
    "salary", "confidential", "trade secret", "secret", "ssn", "social security",
    "privileged", "classified", "internal use only"
]

def detect_sensitivity_for_chunk(text: str, metadata: Dict[str, Any]) -> str:
    t = text.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in t:
            return "high"
    return "low"

def role_allows(role: str, sensitivity: str) -> bool:
    """
    Simple RBAC: low_rank cannot see 'high' sensitive chunks
    """
    role = role.lower()
    if sensitivity == "high" and role != "high_rank" and role != "admin":
        return False
    return True
