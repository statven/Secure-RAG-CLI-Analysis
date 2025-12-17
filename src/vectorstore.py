# src/vectorstore.py
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


from langchain_core.documents  import Document
import os
import pickle
from typing import List, Dict, Any

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # быстрый вариант

class VectorStoreClient:
    def __init__(self, index_dir: str = "data/faiss_index"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self._load_index()

    def _load_index(self):
        idx_path = f"{self.index_dir}/index.pkl"
        if os.path.exists(idx_path):
            with open(idx_path, "rb") as f:
                self.vs = pickle.load(f)
        else:
            # пустой FAISS
            self.vs = None

    def _save_index(self):
        idx_path = f"{self.index_dir}/index.pkl"
        with open(idx_path, "wb") as f:
            pickle.dump(self.vs, f)

    def add_documents(self, doc_id: str, chunks: List[Dict[str, Any]]):
        docs = []
        metadatas = []
        texts = []
        for c in chunks:
            texts.append(c["text"])
            metadatas.append(c["metadata"])
        if self.vs is None:
            self.vs = FAISS.from_texts(texts, self.emb, metadatas=metadatas)
        else:
            self.vs.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)])
        self._save_index()

    def query(self, q: str, k: int = 3, metadata_filter: dict = None):
        if self.vs is None:
            return []
        return self.vs.similarity_search(q, k=k, filter=metadata_filter or {})
