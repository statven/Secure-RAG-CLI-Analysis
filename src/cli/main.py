"""Typer-based CLI entrypoint (placeholder)."""
import typer
from rich import print

app = typer.Typer(help="Secure Context-Aware CLI RAG")

@app.command()
def query(
    file: str = typer.Option(..., help="Path to document"),
    role: str = typer.Option("low_rank", help="Role: low_rank|high_rank|admin"),
    q: str = typer.Argument(..., help="Natural language query")
):
    """Single-shot query against a document (placeholder)."""
    print(f"[bold cyan]Querying[/] file={file} role={role} q={q}")
    print("[yellow]This is a placeholder — implement rag_engine and ingestion pipelines.[/]")
