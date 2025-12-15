#!/usr/bin/env python3
"""
bootstrap_project.py

Create a scaffold for the 'secure-cli-rag' project.

Usage:
    python bootstrap_project.py            # create in ./secure-cli-rag
    python bootstrap_project.py --path /tmp/myproject --overwrite

Requirements: Python 3.10+
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import textwrap

ROOT_DEFAULT = "secure-cli-rag"


FILES = {
    ".gitignore": textwrap.dedent("""\
        __pycache__/
        .env
        .env.*
        venv/
        .venv/
        .DS_Store
        .idea/
        .vscode/
        *.pyc
        /dist
        /build
        *.sqlite3
        """),
    "README.md": textwrap.dedent("""\
        # secure-cli-rag

        Secure Context-Aware CLI RAG — scaffold repository.

        ## Quickstart
        1. Create virtualenv:
           `python -m venv .venv && source .venv/bin/activate`
        2. Install dependencies:
           `pip install -r requirements.txt`
        3. Run CLI (placeholder):
           `python rag.py query --file data/contracts/example.pdf --role low_rank \"What is the timeline?\"`

        TODO: fill out documentation, architecture diagrams and examples in /docs.
        """),
    "requirements.txt": textwrap.dedent("""\
        typer
        rich
        langchain
        faiss-cpu
        pandas
        openpyxl
        python-docx
        PyMuPDF
        sentence-transformers
        pytest
        """),
    "Dockerfile": textwrap.dedent("""\
        # Basic Dockerfile placeholder
        FROM python:3.11-slim
        WORKDIR /app
        COPY . /app
        RUN pip install -r requirements.txt
        CMD [\"python\", \"rag.py\"]
        """),
    "rag.py": textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"Top-level launcher for the Secure CLI RAG.\"\"\"
        import sys
        try:
            from src.cli.main import app
        except Exception as e:
            print(\"Cannot import CLI app (placeholder). Please implement src/cli/main.py.\")
            raise

        if __name__ == '__main__':
            # Example: python rag.py query --file path --role low_rank "What is X?"
            sys.exit(app())
        """),
    ".devcontainer/devcontainer.json": textwrap.dedent("""\
        {
          "name": "secure-cli-rag (devcontainer)",
          "image": "mcr.microsoft.com/vscode/devcontainers/python:3.11",
          "workspaceFolder": "/workspace"
        }
        """),
    "docs/architecture.md": "# Architecture\n\nHigh-level architecture diagram and notes.\n\nTODO: add diagrams.",
    "docs/README.md": "# Documentation\n\nProject documentation goes here."
}

# minimal source placeholders
SRC_FILES = {
    "src/cli/__init__.py": '"""CLI package for secure-cli-rag"""',
    "src/cli/main.py": textwrap.dedent("""\
        \"\"\"Typer-based CLI entrypoint (placeholder).\"\"\"
        import typer
        from rich import print

        app = typer.Typer(help=\"Secure Context-Aware CLI RAG\")

        @app.command()
        def query(
            file: str = typer.Option(..., help=\"Path to document\"),
            role: str = typer.Option(\"low_rank\", help=\"Role: low_rank|high_rank|admin\"),
            q: str = typer.Argument(..., help=\"Natural language query\")
        ):
            \"\"\"Single-shot query against a document (placeholder).\"\"\"
            print(f\"[bold cyan]Querying[/] file={file} role={role} q={q}\")
            print(\"[yellow]This is a placeholder — implement rag_engine and ingestion pipelines.[/]\")
        """),
    "src/core/__init__.py": '"""Core RAG logic package"""',
    "src/core/llm.py": textwrap.dedent("""\
        \"\"\"LLM provider abstraction (placeholder).\"\"\"
        # TODO: implement provider-agnostic wrappers for LLMs (Gemini/GPT/Llama)
        def generate(prompt: str) -> str:
            return \"[LLM output placeholder]\"
        """),
    "src/core/embeddings.py": textwrap.dedent("""\
        \"\"\"Embeddings abstraction (placeholder).\"\"\"
        # TODO: connect to sentence-transformers or HF embeddings API
        def embed_texts(texts):
            return [[0.0]] * len(texts)
        """),
    "src/core/rag_engine.py": textwrap.dedent("""\
        \"\"\"RAG orchestration placeholder.\"\"\"
        # TODO: implement retrieval + prompt composition + citation enforcement
        def answer_query(query, retriever, role):
            return {\"answer\": \"Placeholder answer\", \"citations\": []}
        """),
    "src/ingestion/__init__.py": '"""Ingestion package"""',
    "src/ingestion/ingestion.py": textwrap.dedent("""\
        \"\"\"Top-level ingestion orchestrator (placeholder).\"\"\"
        # TODO: implement file type detection and dispatch to parsers
        def ingest(path):
            return {\"status\": \"not implemented\"}
        """),
    "src/ingestion/parsers/__init__.py": '"""Parsers package"""',
    "src/ingestion/parsers/pdf_parser.py": textwrap.dedent("""\
        \"\"\"PDF parser placeholder using PyMuPDF or PyPDF2.\"\"\"
        def parse(path):
            return [{\"page\": 1, \"text\": \"Example text\"}]
        """),
    "src/ingestion/parsers/docx_parser.py": textwrap.dedent("""\
        \"\"\"DOCX parser placeholder using python-docx.\"\"\"
        def parse(path):
            return [{\"paragraph\": 1, \"text\": \"Example paragraph\"}]
        """),
    "src/ingestion/parsers/txt_parser.py": textwrap.dedent("""\
        \"\"\"TXT parser placeholder.\"\"\"
        def parse(path):
            with open(path, 'r', encoding='utf-8') as f:
                return [{\"text\": f.read()}]
        """),
    "src/ingestion/parsers/csv_parser.py": textwrap.dedent("""\
        \"\"\"CSV parser placeholder using pandas.\"\"\"
        def parse(path):
            # TODO: implement row->chunk conversion with metadata (row/col)
            return []
        """),
    "src/ingestion/parsers/excel_parser.py": textwrap.dedent("""\
        \"\"\"Excel parser placeholder using openpyxl/pandas.\"\"\"
        def parse(path):
            return []
        """),
    "src/security/__init__.py": '"""Security package (RBAC & redaction)"""',
    "src/security/access_control.py": textwrap.dedent("""\
        \"\"\"Role definitions and policy mapping.\"\"\"
        ROLES = (\"low_rank\", \"high_rank\", \"admin\")

        # Sensitivity levels map to allowed roles
        SENSITIVITY_POLICY = {
            \"public\": ROLES,
            \"internal\": (\"high_rank\", \"admin\"),
            \"secret\": (\"admin\",)
        }

        def allowed(role: str, sensitivity: str) -> bool:
            return role in SENSITIVITY_POLICY.get(sensitivity, ())
        """),
    "src/security/security.py": textwrap.dedent("""\
        \"\"\"Security & redaction helpers.\"\"\"
        from .access_control import allowed

        def filter_chunks(chunks, role):
            \"\"\"Return only chunks allowed for the role.\"\"\"
            out = []
            for c in chunks:
                sensitivity = c.get('sensitivity', 'public')
                if allowed(role, sensitivity):
                    out.append(c)
            return out
        """)
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, overwrite: bool = False) -> bool:
    """
    Write file with content; do not overwrite unless overwrite=True.
    Returns True if file was written, False if skipped.
    """
    if path.exists() and not overwrite:
        return False
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
    return True


def create_structure(root: Path, overwrite: bool = False) -> None:
    # directories to create
    dirs = [
        ".devcontainer",
        "data/contracts",
        "data/tabular",
        "docs",
        "src/cli",
        "src/core",
        "src/ingestion",
        "src/ingestion/parsers",
        "src/security",
    ]
    for d in dirs:
        ensure_dir(root.joinpath(d))

    # create files from FILES dict
    for rel, content in FILES.items():
        p = root.joinpath(rel)
        written = write_file(p, content, overwrite=overwrite)
        print(f"{'[WROTE]' if written else '[SKIP ]'} {p}")

    # create src files
    for rel, content in SRC_FILES.items():
        p = root.joinpath(rel)
        written = write_file(p, content, overwrite=overwrite)
        print(f"{'[WROTE]' if written else '[SKIP ]'} {p}")

    # create example placeholder files in data/
    example_pdf = root.joinpath("data/contracts/example.pdf")
    if not example_pdf.exists() or overwrite:
        ensure_dir(example_pdf.parent)
        # create small text file named .placeholder instead of binary PDF
        example_pdf.write_text("Placeholder for example.pdf (add a real PDF here)", encoding="utf-8")
        print(f"[WROTE] {example_pdf}")

    example_csv = root.joinpath("data/tabular/example.csv")
    if not example_csv.exists() or overwrite:
        ensure_dir(example_csv.parent)
        example_csv.write_text("id,name,amount\n1,Example,100\n", encoding="utf-8")
        print(f"[WROTE] {example_csv}")

    # make top-level rag.py executable if created
    rag_path = root.joinpath("rag.py")
    if rag_path.exists():
        try:
            current_mode = rag_path.stat().st_mode
            # set executable for user
            rag_path.chmod(current_mode | 0o755)
            print(f"[CHMOD] made {rag_path} executable")
        except Exception as e:
            print(f"[WARN] could not chmod rag.py: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap secure-cli-rag project scaffold.")
    parser.add_argument("--path", "-p", default=ROOT_DEFAULT, help="Root path for scaffold.")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        root.mkdir(parents=True)
        print(f"[MKDIR] Created project root: {root}")
    else:
        print(f"[INFO] Using existing path: {root}")

    create_structure(root, overwrite=args.overwrite)
    print("\nDone. Project scaffold created at:", root)
    print("Next steps:")
    print("  - create and activate venv")
    print("  - install dependencies: pip install -r requirements.txt")
    print("  - implement pipeline: ingestion -> embeddings -> vectorstore -> rag_engine -> security filters")


if __name__ == "__main__":
    main()
