from typing import Dict, Any


LEVEL_LOW = "low"
LEVEL_HIGH = "high"
VALID_LEVELS = [LEVEL_LOW, LEVEL_HIGH]

def validate_sensitivity(level: str) -> str:
    """Checks that the entered level is correct."""
    level = level.lower().strip()
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid sensitivity level: '{level}'. Allowed: {VALID_LEVELS}")
    return level

def role_allows(user_role: str, chunk_sensitivity: str) -> bool:
    """
    RBAC Logic:
    - Admin/High_rank sees everything
    - Low_rank only sees 'low'.
    """
    user_role = user_role.lower()
    chunk_sensitivity = chunk_sensitivity.lower()

    # public chunck
    if chunk_sensitivity == LEVEL_LOW:
        return True
    
    # secret
    if chunk_sensitivity == LEVEL_HIGH:
        return user_role in ["high_rank", "admin"]
    
    return False