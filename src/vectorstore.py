import os
import shutil
import uuid 
from datetime import datetime 
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class VectorStoreClient:
    def __init__(self, index_dir: str = "data/faiss_index"):
        self.index_dir = index_dir
        self.emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vs = self._load_index()

    def _load_index(self):
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            return FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
        return None

    def _save_index(self):
        if self.vs:
            self.vs.save_local(self.index_dir)

    def upsert_document(self, doc_id: str, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # 1. If the index exists, delete old chunks
        if self.vs is not None:

            ids_to_delete = []
            for uid, doc in self.vs.docstore._dict.items():
                if doc.metadata.get("doc_id") == doc_id:
                    ids_to_delete.append(uid)
            
            if ids_to_delete:
                self.vs.delete(ids_to_delete)
                print(f"Upsert: Deleted {len(ids_to_delete)} old chunks for '{doc_id}'")

        # 2. Adding new
        if self.vs is None:
            self.vs = FAISS.from_texts(texts, self.emb, metadatas=metadatas)
        else:
            self.vs.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)])
        
        self._save_index()
    def add_memory_trace(self, text: str, role: str):
        """
        Indexes a piece of conversation history as a retrievable memory trace.
        This enables 'Long-Term Memory' for the DH-RAG system.
        """
        if not text: return

        meta = {
            "doc_id": f"memory_{uuid.uuid4().hex[:8]}",
            "source_type": "episodic_memory", 
            "created_at": str(datetime.now()),
            "sensitivity": "low", 
            "role_context": role
        }
        
        # Embed and add immediately
        if self.vs is None:
            self.vs = FAISS.from_texts([text], self.emb, metadatas=[meta])
        else:
            self.vs.add_texts([text], metadatas=[meta])
            
        self._save_index()

    def get_hybrid_retriever(self, k: int = 30):
        if self.vs is None:
            return None
            
        # 1. Semantic Retriever (FAISS)
        faiss_retriever = self.vs.as_retriever(search_kwargs={"k": k})
        
        # 2. Keyword Retriever (BM25)
        all_docs = list(self.vs.docstore._dict.values())
        if not all_docs:
            return faiss_retriever # Fallback if empty
            
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
        
        # 3. Ensemble (Weights: 0.5 semantic, 0.5 keyword)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7] # More weight on FAISS because queries are still vague; BM25 stays as a keyword safety net
        )
        return ensemble_retriever
    