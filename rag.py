#!/usr/bin/env python3
"""Top-level launcher for the Secure CLI RAG."""
import sys
try:
    from src.cli.main import app
except Exception as e:
    print("Cannot import CLI app (placeholder). Please implement src/cli/main.py.")
    raise

if __name__ == '__main__':
    # Example: python rag.py query --file path --role low_rank "What is X?"
    sys.exit(app())
