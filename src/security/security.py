"""Security & redaction helpers."""
from .access_control import allowed

def filter_chunks(chunks, role):
    """Return only chunks allowed for the role."""
    out = []
    for c in chunks:
        sensitivity = c.get('sensitivity', 'public')
        if allowed(role, sensitivity):
            out.append(c)
    return out
