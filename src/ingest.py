# src/ingest.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# optional libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx
except Exception:
    docx = None

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore import VectorStoreClient
from src.security import detect_sensitivity_for_chunk

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed (pip install pymupdf)")
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    return pages

def extract_text_from_docx(path: str) -> List[Dict[str, Any]]:
    if docx is None:
        raise RuntimeError("python-docx not installed (pip install python-docx)")
    d = docx.Document(path)
    full = "\n".join(p.text for p in d.paragraphs)
    return [{"page": 1, "text": full}]

def extract_from_txt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    return [{"page": 1, "text": t}]

def extract_from_table(path: str) -> List[Dict[str, Any]]:
    # CSV or Excel -> convert rows into text chunks with row metadata
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    rows = []
    for i, row in df.iterrows():
        rows.append({"page": 1, "row": int(i)+1, "text": row.to_json()})
    return rows

def chunk_texts(pages: List[Dict[str, Any]], doc_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for p in pages:
        raw_text = p.get("text", "")
        parts = splitter.split_text(raw_text)
        for idx, part in enumerate(parts):
            metadata = {
                "doc_id": doc_id,
                "page": p.get("page"),
                "chunk_id": f"{p.get('page')}-{idx}",
            }
            # detect sensitivity flag
            metadata["sensitivity"] = detect_sensitivity_for_chunk(part, metadata)
            chunks.append({"text": part, "metadata": metadata})
    return chunks

def ingest_file(path: str, doc_id: str):
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        pages = extract_text_from_pdf(path)
    elif ext == ".docx":
        pages = extract_text_from_docx(path)
    elif ext in [".txt"]:
        pages = extract_from_txt(path)
    elif ext in [".csv", ".xlsx", ".xls"]:
        pages = extract_from_table(path)
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")

    chunks = chunk_texts(pages, doc_id)
    vs = VectorStoreClient()
    vs.add_documents(doc_id, chunks)
    # persist basic manifest
    man_path = Path("data") / f"{doc_id}_manifest.json"
    man_path.parent.mkdir(exist_ok=True)
    man_path.write_text(json.dumps({"doc_id": doc_id, "num_chunks": len(chunks)}), encoding="utf-8")
