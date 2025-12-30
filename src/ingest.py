# src/ingest.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# optional libs
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
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed (pip install pymupdf)")
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text, "source_type": "pdf"})
    return pages

def extract_text_from_docx(path: str) -> List[Dict[str, Any]]:

    if docx is None:
        raise RuntimeError("python-docx not installed (pip install python-docx)")
    d = docx.Document(path)
    # Группируем параграфы, чтобы не дробить слишком мелко, но сохранять контекст
    pages = []
    current_chunk = []
    current_length = 0
    page_counter = 1
    
    for p in d.paragraphs:
        if not p.text.strip():
            continue
        current_chunk.append(p.text)
        current_length += len(p.text)
        # Эвристика: каждые ~2000 символов считаем новой "логической страницей" для цитирования
        if current_length > 2000:
            pages.append({"page": page_counter, "text": "\n".join(current_chunk), "source_type": "docx"})
            current_chunk = []
            current_length = 0
            page_counter += 1
            
    if current_chunk:
        pages.append({"page": page_counter, "text": "\n".join(current_chunk), "source_type": "docx"})
        
    return pages

def extract_from_table(path: str) -> List[Dict[str, Any]]:
    """
    Профессиональная обработка таблиц:
    Конвертирует строки в Markdown-формат, сохраняя заголовки колонок.
    Это значительно повышает качество ответов LLM по табличным данным.
    """
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    
    # NaN clear
    df = df.fillna("")
    
    rows = []

    for i, row in df.iterrows():
        #  "| Column1: Value | Column2: Value |"
        row_content = " | ".join([f"{col}: {val}" for col, val in row.items()])
        
        # metadata
        rows.append({
            "page": 1, # for tables always 1
            "row": int(i)+1, 
            "text": row_content,
            "source_type": "table"
        })
    return rows

def chunk_texts(pages: List[Dict[str, Any]], doc_id: str, file_sensitivity: str):

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
                "source_type": p.get("source_type", "unknown"),
                "row_idx": p.get("row", None)  
            }
            
            metadata["sensitivity"] = file_sensitivity
            
            # None clear
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            chunks.append({"text": part, "metadata": metadata})
            
    return chunks

def ingest_file(path: str, doc_id: str, sensitivity: str):
    path_obj = Path(path)
    if not path_obj.exists():
         raise FileNotFoundError(f"{path} does not exist.")
         
    ext = path_obj.suffix.lower()
    if ext == ".pdf":
        pages = extract_text_from_pdf(str(path))
    elif ext == ".docx":
        pages = extract_text_from_docx(str(path))
    elif ext in [".txt", ".md"]: # Added markdown support
        with open(path, "r", encoding="utf-8") as f:
            pages = [{"page": 1, "text": f.read(), "source_type": "text"}]
    elif ext in [".csv", ".xlsx", ".xls"]:
        pages = extract_from_table(str(path))
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")
    clean_sensitivity = validate_sensitivity(sensitivity)
    chunks = chunk_texts(pages, doc_id, file_sensitivity=clean_sensitivity)
    
    vs = VectorStoreClient()
    vs.add_documents(doc_id, chunks)
    
    # Сохраняем манифест с информацией об уровне доступа
    man_path = Path("data") / f"{doc_id}_manifest.json"
    man_path.parent.mkdir(exist_ok=True)
    man_path.write_text(json.dumps({
        "doc_id": doc_id, 
        "sensitivity": clean_sensitivity,
        "num_chunks": len(chunks)
    }), encoding="utf-8")