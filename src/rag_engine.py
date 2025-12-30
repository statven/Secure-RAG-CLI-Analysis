# src/rag_engine.py
import os
from typing import Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI # Google Gemini Integration
from src.vectorstore import VectorStoreClient

class RagEngine:
    def __init__(self, role: str = "low_rank", model_type: str = "flash"):
        self.role = role
        self.vs = VectorStoreClient()
        
        # Выбор модели согласно ТЗ
        # Flash - для быстрых ответов (дефолт)
        # Pro - для сложных рассуждений
        model_name = "gemini-1.5-pro" if model_type == "pro" else "gemini-2.5-flash"
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            convert_system_message_to_human=True # Иногда требуется для Gemini
        )

    def _get_security_filter(self) -> Dict:
        """
        RBAC Pre-filtering: 
        Low Rank пользователи физически не могут получить чанки с sensitivity='high'.
        """
        if self.role.lower() in ["high_rank", "admin"]:
            return {} 
        else:
            return {"sensitivity": "low"}

    def answer(self, question: str, doc_id: Optional[str] = None) -> str:
        # 1. Сбор фильтров
        filters = self._get_security_filter()
        if doc_id:
            filters["doc_id"] = doc_id

        # 2. Поиск (Retrieval)
        retrieved_docs = self.vs.query(question, k=10, metadata_filter=filters)

        if not retrieved_docs:
            return "ACCESS DENIED OR INFO NOT FOUND: No accessible documents contain the answer."

        # 3. Формирование контекста
        context_parts = []
        citations_map = []
        
        for i, d in enumerate(retrieved_docs):
            meta = d.metadata
            # Умное цитирование: Row для таблиц, Page для документов
            if "row_idx" in meta:
                ref = f"Doc: {meta.get('doc_id')} | Row: {meta.get('row_idx')}"
            else:
                ref = f"Doc: {meta.get('doc_id')} | Page: {meta.get('page')}"
            
            clean_text = d.page_content.replace("\n", " ").strip()
            context_parts.append(f"Source [{i+1}] ({ref}): {clean_text}")
            citations_map.append(f"[{i+1}] {ref}")

        # 4. Генерация (Generation)
        prompt = self._build_prompt(question, context_parts)
        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            return f"Error calling Google Gemini: {str(e)}"

        return f"{response}\n\n" + "-"*30 + "\nREFERENCES:\n" + "\n".join(citations_map)

    def _build_prompt(self, question: str, context_parts: List[str]):
        ctx_str = "\n\n".join(context_parts)
        system_msg = (
            "You are a secure corporate analysis assistant. "
            "Answer strictly based on the provided CONTEXT. "
            "If the answer is not in the context, say 'Information not found'.\n"
            "MANDATORY: Cite sources using brackets like [1] or [2] for every claim."
        )
        return [
            ("system", system_msg),
            ("human", f"CONTEXT:\n{ctx_str}\n\nQUESTION: {question}")
        ]