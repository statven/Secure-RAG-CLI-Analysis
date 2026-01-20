import os
import shutil
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss

# Parameters
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "data/storage"

class VectorStoreClient:
    def __init__(self):
        self.persist_dir = PERSIST_DIR
        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> VectorStoreIndex:
        """Loads the LlamaIndex from disk or creates a new one."""
        if os.path.exists(self.persist_dir):
            try:
                # Load existing storage
                vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=self.persist_dir
                )
                return load_index_from_storage(storage_context, embed_model=self.embed_model)
            except Exception as e:
                print(f"Error loading index, creating new: {e}")
        
        # Create new FAISS index
        # 384 dimensions for all-MiniLM-L6-v2
        d = 384 
        faiss_index = faiss.IndexFlatIP(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, embed_model=self.embed_model
        )
        return index

    def _save_index(self):
        """Persists the index to disk."""
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def upsert_document(self, doc_id: str, nodes: List[Document]):
        """
        A safe upsert operation for FAISS.
        """
        try:
            # FAISS may not support deletion.. 

            self.index.delete_ref_doc(doc_id, delete_from_docstore=True)
        except NotImplementedError:
            print(f"Warning: FAISS does not support deleting {doc_id}. The data will be added as new.")
        
        #  Inserting new nodes
        self.index.insert_nodes(nodes)
        
        # saving
        self._save_index()

    def add_memory_trace(self, text: str, role: str):
        """
        Adds episodic memory.
        """
        import uuid
        from datetime import datetime
        
        if not text: return

        doc = Document(
            text=text,
            metadata={
                "doc_id": f"memory_{uuid.uuid4().hex[:8]}",
                "source_type": "episodic_memory",
                "created_at": str(datetime.now()),
                "sensitivity": "low",
                "role_context": role
            }
        )
        # We don't link memory traces to a specific ref_doc_id in the same way as files,
        # or we treat each memory as a unique doc.
        self.index.insert(doc)
        self._save_index()
    
    def get_retriever(self, k: int = 30):
        """Returns a retriever from the index."""
        return self.index.as_retriever(similarity_top_k=k)