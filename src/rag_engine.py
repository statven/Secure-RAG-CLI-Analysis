# src/rag_engine.py
from typing import Optional
from src.vectorstore import VectorStoreClient
from src.security import role_allows
import textwrap

class RagEngine:
    def __init__(self, role: str = "low_rank"):
        self.role = role
        self.vs = VectorStoreClient()

    def _filter_retrieved(self, docs):
        # docs are LangChain Document objects with metadata
        kept = []
        for d in docs:
            sens = d.metadata.get("sensitivity", "low")
            if role_allows(self.role, sens):
                kept.append(d)
        return kept

    def answer(self, question: str, doc_id: Optional[str] = None) -> str:
        # apply doc filter to only search in doc_id if provided
        meta_filter = {"doc_id": doc_id} if doc_id else None
        retrieved = self.vs.query(question, k=5, metadata_filter=meta_filter)
        filtered = self._filter_retrieved(retrieved)
        if not filtered:
            return "Information not found."

        # build context with exact citations
        context_parts = []
        for d in filtered:
            meta = d.metadata
            citation = f"Doc:{meta.get('doc_id')} Page:{meta.get('page')} Chunk:{meta.get('chunk_id')}"
            excerpt = textwrap.shorten(d.page_content.replace("\n", " "), width=400, placeholder="...")
            context_parts.append(f"[{citation}] {excerpt}")

        prompt = self._build_prompt(question, context_parts)
        # call LLM - here we use a simple deterministic summarizer (placeholder)
        # Replace with real LLM call in production
        answer = self._dummy_llm(prompt)
        # attach citations (we built context so LLM can reference)
        return answer + "\n\nCitations:\n" + "\n".join([p.split("] ")[0] + "]" for p in context_parts])

    def _build_prompt(self, question, context_parts):
        ctx = "\n\n".join(context_parts)
        prompt = (
            "You are an assistant that MUST answer using ONLY the provided CONTEXT. "
            "If answer cannot be found in context, reply exactly: Information not found.\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer concisely and include precise citation markers."
        )
        return prompt

    def _dummy_llm(self, prompt: str) -> str:
        # Simple rule: if any keyword from question appears in context, return a synthetic answer
        # This is placeholder for actual LLM. Replace this with provider call.
        if "timeline" in prompt.lower():
            return "The timeline is described in the context chunks. (See citations below)"
        return "Information not found."
