# src/vectorstore.py
import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings # Updated import standard
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Используем модель, указанную в ТЗ (быстрая и эффективная)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class VectorStoreClient:
    def __init__(self, index_dir: str = "data/faiss_index"):
        self.index_dir = index_dir
        # Используем HuggingFaceEmbeddings (CPU optimized)
        self.emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vs = self._load_index()

    def _load_index(self):
        """Загружает индекс FAISS используя нативный метод load_local."""
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            # allow_dangerous_deserialization нужен для локальных доверенных файлов
            return FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
        else:
            return None

    def _save_index(self):
        """Сохраняет индекс используя нативный метод save_local."""
        if self.vs:
            self.vs.save_local(self.index_dir)


    def add_documents(self, doc_id: str, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # 1. Если индекс уже существует, пытаемся удалить старые записи этого doc_id
        if self.vs is not None:
            try:
                # В новых версиях LangChain FAISS поддерживает delete по фильтру
                self.vs.delete(filter={"doc_id": doc_id})
            except Exception:
                # Если delete не поддерживается напрямую, в локальных проектах 
                # проще очистить и пересобрать индекс для чистоты
                pass

        # 2. Добавление новых документов
        if self.vs is None:
            self.vs = FAISS.from_texts(texts, self.emb, metadatas=metadatas)
        else:
            self.vs.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)])
        
        self._save_index()

    def query(self, q: str, k: int = 10, metadata_filter: dict = None):
        if self.vs is None:
            return []
        
        # КЛЮЧЕВОЕ: Если фильтр пустой (для роли high_rank), не передаем его, 
        # чтобы FAISS искал по всей базе
        if not metadata_filter:
            return self.vs.similarity_search(q, k=k)
            
        return self.vs.similarity_search(q, k=k, filter=metadata_filter)