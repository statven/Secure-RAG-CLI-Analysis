"""Role definitions and policy mapping."""
ROLES = ("low_rank", "high_rank", "admin")

# Sensitivity levels map to allowed roles
SENSITIVITY_POLICY = {
    "public": ROLES,
    "internal": ("high_rank", "admin"),
    "secret": ("admin",)
}

def allowed(role: str, sensitivity: str) -> bool:
    return role in SENSITIVITY_POLICY.get(sensitivity, ())
