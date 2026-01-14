import os
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from sentence_transformers import util

from src.vectorstore import VectorStoreClient
from src.security import role_allows

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

class RagEngine:
    def __init__(self, role: str = "low_rank", model_type: str = "flash"):
        self.role = role
        self.vs = VectorStoreClient()
        self.console = Console()
        
        # 1. Active Context (Structured State)
        # Instead of just a text log, we treat summary as the "Active Working Memory".
        self.active_context = "" 
        
        # 2. Last Intent (The "Predicate")
        # Explicitly tracking what the user is looking for (e.g., "counting friends", "dates").
        self.current_intent = "General inquiry"
        
        self.history_file = Path("data/chat_session.json")
        self._load_session() 
        
        main_model_name = "gemini-2.5-flash"
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY env var missing")

        self.llm = ChatGoogleGenerativeAI(
            model=main_model_name,
            temperature=0,
        )

    def _load_session(self):
        """Loads the active context state."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                self.active_context = data.get("summary", "")
                self.current_intent = data.get("intent", "General inquiry")
            except: pass

    def _save_session(self):
        """Persists state."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.active_context,
            "intent": self.current_intent
        }
        self.history_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _rewrite_query(self, raw_query: str) -> str:
        """
        CRITICAL FIX FOR INTENT DRIFT.
        Converts ambiguous follow-ups ("And him?") into standalone queries ("Does Voldemort have friends?").
        Uses the Active Context to resolve references and carry over the predicate.
        """
        if not self.active_context:
            return raw_query

        # Prompt explicitly asks to merge context + query into a Standalone Question
        prompt = (
            "You are a Query Reformulator. Your goal is to make the user's question STANDALONE.\n"
            "Use the Context to fill in missing entities or intents (what is being asked).\n"
            "If the user asks 'And him?', apply the previous question's intent to the new entity.\n\n"
            f"Active Context: {self.active_context}\n"
            f"Last Intent: {self.current_intent}\n"
            f"Raw User Input: {raw_query}\n\n"
            "Standalone Question:"
        )
        
        try:
            # We use a fast call here.
            rewritten = self.llm.invoke(prompt).content.strip()
            # Visual debug for the user to see the "Thinking" process
            self.console.print(f"[dim]🔎 Logic: '{raw_query}' -> '{rewritten}'[/dim]")
            return rewritten
        except:
            return raw_query

    def _update_state(self, last_q: str, last_a: str, rewritten_q: str):
        """
        Updates the Active Context and extracts the Intent.
        Fixes 'Summary Bloat' by aggressively discarding old topics.
        """
        
        prompt = (
            "You are a State Manager. Update the 'Active Context' and 'Current Intent'.\n"
            "1. Active Context: Summarize ONLY the current topic. Discard old, finished topics. Be concise.\n"
            "2. Current Intent: What is the user primarily looking for? (e.g., 'listing friends', 'comparing dates', 'asking biography').\n\n"
            f"Old Context: {self.active_context}\n"
            f"Most Recent Interaction -> Q: {rewritten_q} | A: {last_a}\n\n"
            "Output JSON format: {\"context\": \"...\", \"intent\": \"...\"}"
        )
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            # Basic cleanup to ensure JSON parsing
            clean_json = response.replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            
            self.active_context = data.get("context", "")
            self.current_intent = data.get("intent", "General inquiry")
            
            self.console.print(f"[dim]🧠 Memory: {self.active_context} | Intent: {self.current_intent}[/dim]")
        except Exception as e:
            # Fallback if JSON fails
            self.active_context = f"Last discussed: {rewritten_q}"

    def _retrieve_long_term_memory(self, query: str) -> str:
        """Retrieves episodic memory from Vector Store."""
        if not self.vs: return ""
        retriever = self.vs.get_hybrid_retriever(k=3)
        if not retriever: return ""
        try:
            docs = retriever.invoke(query)
            memories = [d.page_content for d in docs if d.metadata.get("source_type") == "episodic_memory"]
            return "\n".join(memories) if memories else ""
        except: return ""

    def _calculate_triad_metrics(self, question: str, answer: str, context_chunks: list) -> dict:
        """Deterministic RAG Triad Metrics."""
        if not context_chunks: return {}
        q_emb = self.vs.emb.embed_query(question)
        a_emb = self.vs.emb.embed_query(answer)
        c_embs = [self.vs.emb.embed_query(c) for c in context_chunks[:3]]
        ctx_rel = np.mean([util.cos_sim(q_emb, c).item() for c in c_embs]) if c_embs else 0
        ans_rel = util.cos_sim(q_emb, a_emb).item()
        return {"context_rel": round(max(0, ctx_rel), 2), "answer_rel": round(max(0, ans_rel), 2)}

    def answer(self, question: str, doc_id: Optional[str] = None):
        t_start = time.time()
        
        # --- STEP 1: Query Transformation (Fixes Intent Drift) ---
        # Instead of searching for "And Voldemort", we search for "Who are Voldemort's friends?"
        standalone_query = self._rewrite_query(question)
        
        # --- STEP 2: Retrieval (using Standalone Query) ---
        # Now retrieval is focused on the PREDICATE (friends), not just the ENTITY (Voldemort).
        retriever = self.vs.get_hybrid_retriever(k=15)
        raw_docs = retriever.invoke(standalone_query) if retriever else []
        
        allowed_docs = [d for d in raw_docs if role_allows(self.role, d.metadata.get("sensitivity", "low"))]
        past_memories = self._retrieve_long_term_memory(standalone_query)

        # Prepare Context
        context_text = ""
        citations = []
        chunks_for_metric = []
        for i, d in enumerate(allowed_docs[:10]):
            clean = " ".join(d.page_content.split())
            context_text += f"Source [{i+1}]: {clean}\n\n"
            chunks_for_metric.append(clean)
            citations.append({"id": i+1, "file": d.metadata.get('doc_id'), "loc": f"Pg {d.metadata.get('page')}"})

        # --- STEP 3: Generation (Strictly Grounded) ---
        # We explicitly tell the LLM the User's Intent to keep it focused.
        sys_msg = (
            "You are a Strict RAG Assistant. "
            f"Current Intent: {self.current_intent.upper()}.\n" 
            "1. Answer ONLY what is asked based on Documents.\n"
            "2. If the documents describe the entity but do NOT contain info matching the Intent (e.g., no friends mentioned), "
            "state clearly: 'The documents do not contain information about [Intent] for [Entity].' \n"
            "3. Do NOT provide a generic biography.\n"
        )
        
        user_msg = (
            f"CONTEXT:\n{context_text}\n\n"
            f"MEMORY:\n{past_memories}\n\n"
            f"QUESTION: {standalone_query}" # We feed the rewritten query to the LLM too
        )
        
        try:
            raw_response = self.llm.invoke([("system", sys_msg), ("human", user_msg)]).content
            final_answer = raw_response.strip()
        except Exception as e:
            self.console.print(f"[red]LLM Error: {e}[/red]")
            return

        # --- STEP 4: State Update (Fixes Memory Bloat) ---
        self._update_state(question, final_answer, standalone_query)
        self._save_session()
        
        # Index into Long Term Memory
        self.vs.add_memory_trace(f"Q: {standalone_query}\nA: {final_answer}", self.role)

        # Output
        metrics = self._calculate_triad_metrics(standalone_query, final_answer, chunks_for_metric)
        total_time = round(time.time() - t_start, 2)
        
        self.console.print(Panel(final_answer, title=f"DH-RAG (Intent: {self.current_intent})", border_style="green"))
        
        if citations:
            ref_table = Table(title="Sources", box=box.SIMPLE)
            ref_table.add_column("Ref"); ref_table.add_column("Doc"); ref_table.add_column("Loc")
            for c in citations[:5]:
                ref_table.add_row(f"[{c['id']}]", c['file'], c['loc'])
            self.console.print(ref_table)

        self.console.print(f"[dim]Metrics: CtxRel={metrics.get('context_rel',0)} | AnsRel={metrics.get('answer_rel',0)} | Time={total_time}s[/dim]")