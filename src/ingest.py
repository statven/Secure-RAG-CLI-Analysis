import os
import json
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

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from src.vectorstore import VectorStoreClient
from src.security import validate_sensitivity

CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200 

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
    full_text = []
    for p in d.paragraphs:
        if p.text.strip():
            full_text.append(p.text)
    return [{"page": 1, "text": "\n".join(full_text), "source_type": "docx"}]

def extract_from_table(path: str) -> List[Dict[str, Any]]:
    """Custom logic to convert Table rows into Markdown chunks."""
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df = df.fillna("")

    columns = list(df.columns)
    header = "| " + " | ".join(map(str, columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    chunks = []
    batch_size = 5  

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i+batch_size]
        text_lines = [header, separator]

        for _, row in batch.iterrows():
            row_str = "| " + " | ".join(map(str, row.values)) + " |"
            text_lines.append(row_str)

        chunk_text = "\n".join(text_lines)
        page_no = (i // batch_size) + 1
        
        chunks.append({
            "page": page_no,
            "row_start": i+1,
            "row_end": i+len(batch),
            "text": chunk_text,
            "source_type": "table_markdown"
        })
    return chunks

def create_documents(pages: List[Dict[str, Any]], doc_id: str, file_sensitivity: str) -> List[Document]:
    """
    Splits text and creates LlamaIndex Document objects (Nodes).
    """
    # LlamaIndex splitter
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    docs_to_insert = []
    global_counter = 0

    for p in pages:
        raw_text = (p.get("text") or "").strip()
        if not raw_text:
            continue

        # If it's our custom table markdown, we treat the whole batch as one chunk
        if p.get("source_type") == "table_markdown":
            text_chunks = [raw_text]
        else:
            text_chunks = splitter.split_text(raw_text)

        page_no = p.get("page", 1)

        for part in text_chunks:
            chunk_id = f"{doc_id}_p{page_no}_{global_counter}"
            
            metadata = {
                "doc_id": doc_id,
                "page": page_no,
                "chunk_id": chunk_id,
                "source_type": p.get("source_type", "unknown"),
                "sensitivity": file_sensitivity
            }
            
            if "row_start" in p:
                metadata["row_idx"] = f"{p['row_start']}-{p['row_end']}"

            # Create LlamaIndex Document (which will become a Node in the index)
            # We explicitly set doc_id so we can overwrite it later (upsert logic)
            doc = Document(text=part, metadata=metadata)
            doc.doc_id = chunk_id # Internal ID for node
            
            # Important: set ref_doc_id to the file ID so we can delete the whole file later
            doc.metadata["ref_doc_id"] = doc_id 
            
            docs_to_insert.append(doc)
            global_counter += 1

    return docs_to_insert

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
    
    # Convert to LlamaIndex Documents
    nodes = create_documents(pages, doc_id, clean_sensitivity)
    
    # Upsert to VectorStore
    vs_client = VectorStoreClient()
    vs_client.upsert_document(doc_id, nodes)
    
    # Manifest saving (Metadata)
    manifests_dir = Path("data") / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    man_path = Path("data") / f"{doc_id}_manifest.json"
    man_path.parent.mkdir(exist_ok=True)
    man_path.write_text(json.dumps({
        "doc_id": doc_id, 
        "sensitivity": clean_sensitivity,
        "num_chunks": len(nodes)
    }), encoding="utf-8")