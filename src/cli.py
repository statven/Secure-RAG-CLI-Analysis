import typer
from pathlib import Path
from rich import print
from src.ingest import ingest_file
from src.rag_engine import RagEngine
from src.security import LEVEL_LOW, LEVEL_HIGH

app = typer.Typer(help="Secure Context-Aware CLI RAG (LlamaIndex Edition)")

@app.command()
def ingest(
    file: Path = typer.Option(..., help="Path to document to ingest"),
    doc_id: str = typer.Option(None, help="Unique document ID (default: filename)"),
    sensitivity: str = typer.Option(
        LEVEL_LOW, 
        "--sensitivity", "-s",
        help=f"Security level: '{LEVEL_LOW}' or '{LEVEL_HIGH}'."
    )
):
    """Ingest file."""
    if not file.exists():
        print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)
        
    doc_id = doc_id or file.stem
    try:
        ingest_file(str(file), doc_id=doc_id, sensitivity=sensitivity)
        color = "red" if sensitivity == LEVEL_HIGH else "green"
        print(f"[bold]Ingested:[/bold] {doc_id}")
        print(f"[bold]Security Level:[/bold] [{color}]{sensitivity}[/{color}]")
    except ValueError as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

@app.command()
def chat(
    role: str = typer.Option(..., help="Your role (admin/low_rank)"),
    doc_id: str = typer.Option(None, help="Filter by document ID (not implemented in this view)")
):
    """Interactive chat."""
    try:
        engine = RagEngine(role=role)
        print(f"[bold green]Entering chat mode (Role: {role}). Type 'exit' to leave.[/bold green]")
        
        while True:
            question = typer.prompt("You")
            if question.lower() in ["exit", "quit", "leave"]:
                break
            engine.answer(question, doc_id=doc_id)
    except Exception as e:
        print(f"[red]Init Error:[/red] {e}")

if __name__ == "__main__":
    app()