import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import docx
except ImportError:
    docx = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore import VectorStoreClient
from src.security import validate_sensitivity

CHUNK_SIZE = 1000#(обьяснить почему )
CHUNK_OVERLAP = 200


SEPARATORS = [
    "\n\n",
    "\n",
    "(?<=\. )", 
    " ",
    "",
    "\u2022",  # Bullet point
    "- ",
    "* ",
    r"\d+\.\s" # Numbered lists (1. , 2. )
]

def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text, "source_type": "pdf"})
    return pages

def extract_text_from_docx(path: str) -> List[Dict[str, Any]]:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path)
    pages = []
    full_text = []
    
    #collect all the text so that the splitter can sort out the sub-items itself.
    for p in d.paragraphs:
        if p.text.strip():
            full_text.append(p.text)
            
    # saving the pages as a single stream with a marker
    pages.append({"page": 1, "text": "\n".join(full_text), "source_type": "docx"})
    return pages

def extract_from_table(path: str) -> List[Dict[str, Any]]:

    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df = df.fillna("")
    
    #  Markdown header
    columns = list(df.columns)
    header = "| " + " | ".join(map(str, columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    chunks = []
    batch_size = 5 # grouping 5 rows to preserve the context of neighbors.
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i+batch_size]
        text_lines = [header, separator]
        
        for _, row in batch.iterrows():
            row_str = "| " + " | ".join(map(str, row.values)) + " |"
            text_lines.append(row_str)
            
        chunk_text = "\n".join(text_lines)
        
        chunks.append({
            "page": 1, 
            "row_start": i+1,
            "row_end": i+len(batch),
            "text": chunk_text,
            "source_type": "table_markdown"
        })
    return chunks

def chunk_texts(pages: List[Dict[str, Any]], doc_id: str, file_sensitivity: str):
    # Using RegEx Separators to Save Lists
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        is_separator_regex=True
    )
    
    chunks = []
    for p in pages:
        raw_text = p.get("text", "")

        if p.get("source_type") == "table_markdown":
            parts = [raw_text]
        else:
            parts = splitter.split_text(raw_text)
        
        for idx, part in enumerate(parts):
            metadata = {
                "doc_id": doc_id,
                "page": p.get("page"),
                "chunk_id": f"{doc_id}_{idx}",
                "source_type": p.get("source_type", "unknown"),
                "sensitivity": file_sensitivity
            }
            if "row_start" in p:
                metadata["row_idx"] = f"{p['row_start']}-{p['row_end']}"
                
            chunks.append({"text": part, "metadata": metadata})
            
    return chunks

def ingest_file(path: str, doc_id: str, sensitivity: str):
    path_obj = Path(path)
    if not path_obj.exists():
         raise FileNotFoundError(f"{path} does not exist.")
         
    handlers = {
        ".pdf":  lambda p: extract_text_from_pdf(str(p)),
        ".docx": lambda p: extract_text_from_docx(str(p)),
        ".txt":  lambda p: [{"page": 1, "text": p.read_text(encoding="utf-8"), "source_type": "text"}],
        ".md":   lambda p: [{"page": 1, "text": p.read_text(encoding="utf-8"), "source_type": "text"}],
        ".csv":  lambda p: extract_from_table(str(p)),
        ".xlsx": lambda p: extract_from_table(str(p)),
        ".xls":  lambda p: extract_from_table(str(p)),
    }

    ext = path_obj.suffix.lower()
    
    handler = handlers.get(ext)
    if not handler:
        raise RuntimeError(f"Unsupported extension: {ext}")
    pages = handler(path_obj)

    clean_sensitivity = validate_sensitivity(sensitivity)
    chunks = chunk_texts(pages, doc_id, file_sensitivity=clean_sensitivity)
    
    vs = VectorStoreClient()
    manifests_dir = Path("data") / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    full_text = " ".join([p.get("text","") for p in pages])
    (manifests_dir / f"{doc_id}.txt").write_text(full_text, encoding="utf-8")
    vs.upsert_document(doc_id, chunks) # enw meth upsert
    
    # Saving the manifesto
    man_path = Path("data") / f"{doc_id}_manifest.json"
    man_path.parent.mkdir(exist_ok=True)
    man_path.write_text(json.dumps({
        "doc_id": doc_id, 
        "sensitivity": clean_sensitivity,
        "num_chunks": len(chunks)
    }), encoding="utf-8")