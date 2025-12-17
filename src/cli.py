# src/cli.py
import typer
from pathlib import Path
from rich import print
from src.ingest import ingest_file
from src.rag_engine import RagEngine

app = typer.Typer(help="Secure Context-Aware CLI RAG")

@app.command()
def ingest(file: Path = typer.Option(..., help="Path to document to ingest"),
           doc_id: str = typer.Option(None, help="Optional document id")):
    """Ingest file into local vectorstore (FAISS)"""
    if not file.exists():
        print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)
    doc_id = doc_id or file.name
    ingest_file(str(file), doc_id=doc_id)
    print(f"[green]Ingested {file} as {doc_id}[/green]")

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
