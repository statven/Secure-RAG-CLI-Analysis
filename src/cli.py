# src/cli.py
import typer
from pathlib import Path
from rich import print
from src.ingest import ingest_file
from src.rag_engine import RagEngine
from src.security import LEVEL_LOW, LEVEL_HIGH

app = typer.Typer(help="Secure Context-Aware CLI RAG")

@app.command()
def ingest(
    file: Path = typer.Option(..., help="Path to document to ingest"),
    doc_id: str = typer.Option(None, help="Unique document ID (default: filename)"),
    sensitivity: str = typer.Option(
        LEVEL_LOW, 
        "--sensitivity", "-s",
        help=f"Security level for the ENTIRE file: '{LEVEL_LOW}' (public) or '{LEVEL_HIGH}' (restricted)."
    )
):
    """
    Ingest file with a specific security level.
    Example: ingest --file salaries.pdf --sensitivity high
    """
    if not file.exists():
        print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)
        
    doc_id = doc_id or file.stem # Используем имя файла без расширения как ID
    
    try:
        ingest_file(str(file), doc_id=doc_id, sensitivity=sensitivity)
        
        color = "red" if sensitivity == LEVEL_HIGH else "green"
        print(f"[bold]Ingested:[/bold] {doc_id}")
        print(f"[bold]Security Level:[/bold] [{color}]{sensitivity}[/{color}]")
        
    except ValueError as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

@app.command()
def query(file: Path = typer.Option(None, help="(optional) file to auto-ingest before query"),
          doc_id: str = typer.Option(None, help="document id to query"),
          role: str = typer.Option(..., help="role: low_rank | high_rank"),
          q: str = typer.Argument(..., help="Question to ask")):
    """Query a document via RAG (strictly grounded answers)"""
    # if file provided, ingest
    if file:
        ingest_file(str(file), doc_id=(doc_id or file.name))
    engine = RagEngine(role=role)
    answer = engine.answer(question=q, doc_id=doc_id)
    print(answer)

if __name__ == "__main__":
    app()
