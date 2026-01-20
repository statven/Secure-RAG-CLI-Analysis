import os
import time
import json
import re
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime 
from typing import  Any
from typing import Optional, List, Dict, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL  # Math engine
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
        
        # --- Computational Engine ---
        self.python_repl = PythonREPL()
        
        # --- State Management ---
        self.active_context = "" 
        self.current_intent = "General inquiry"
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_turns = 500  # cap to avoid unbounded growth
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
        """Loads the active context state and chat turns if present."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                self.active_context = data.get("summary", "")
                self.current_intent = data.get("intent", "General inquiry")
                # load chat_history if present (list of dicts)
                ch = data.get("chat_history", [])
                if isinstance(ch, list):
                    # keep only last max_history_turns
                    self.chat_history = ch[-self.max_history_turns:]
            except Exception:
                pass

    def _save_session(self):
        """Persists state: summary, intent and recent chat turns."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.active_context,
            "intent": self.current_intent,
            "chat_history": self.chat_history[-self.max_history_turns:]
        }
        self.history_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def get_session_summary(self, turns: int = 20) -> Dict[str, Any]:
        """
        Return a small structured summary of the most recent chat turns.
        Attempts to use the LLM to produce concise JSON:
        { "summary": "...", "highlights": ["...","..."] }
        If LLM fails, returns an extractive fallback using last turns.
        """
        recent = self.chat_history[-turns:]
        if not recent:
            return {"summary": "", "highlights": []}

        # Build compact context to feed the summarizer (avoid huge tokens)
        example_text = []
        for t in recent:
            q = (t.get("q") or "").replace("\n", " ")
            a = (t.get("a") or "").replace("\n", " ")
            example_text.append(f"Q: {q}\nA: {a}")
        brief = "\n\n".join(example_text)
        # LLM prompt: request strict JSON only
        prompt = (
            "You are a concise session summarizer. Produce a short JSON with two fields:\n"
            "  - summary: one or two sentences describing the central topics discussed recently\n"
            "  - highlights: an array of 3 short bullet points with the most important facts or actions\n"
            "Do NOT include chain-of-thought. Output EXACTLY valid JSON.\n\n"
            "RECENT TURNS:\n"
            f"{brief[:6000]}\n\n"
            "Output JSON:"
        )
        try:
            resp = self.llm.invoke(prompt).content.strip()
            # sanitize and parse
            clean = resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return {
                "summary": data.get("summary", "") if isinstance(data.get("summary", ""), str) else "",
                "highlights": data.get("highlights", []) if isinstance(data.get("highlights", []), list) else []
            }
        except Exception:
            # Fallback: extractive summary - join last 5 turns as highlights
            highlights = []
            for t in recent[-3:]:
                q = (t.get("q") or "").strip()
                a = (t.get("a") or "").strip()
                highlights.append(f"Q: {q} → A: {a}")
            summary = " ".join([h.split("→")[0] for h in recent[-4:]]) if recent else ""
            return {"summary": summary, "highlights": highlights}

    def _rewrite_query(self, raw_query: str) -> str:
        """
        UNIVERSAL CONTEXT RESOLVER.
        Handles Context, Ellipsis, and Meta-Requests.
        """
        if not self.active_context:
            return raw_query

        # 1. META-INTENT CHECK (Intercept questions about the session itself)
        # Regex looks for patterns like "summarize chat", "what did we discuss", "previous dialogue"
        meta_pattern = r"\b(summarize|recap|what.*(discuss|talk)|previous|last)\b.*\b(chat|dialogue|conversation|session)\b"
        if re.search(meta_pattern, raw_query, re.IGNORECASE):
            return "__META_SUMMARIZE_SESSION__"

        # 2. STANDARD REWRITE (Domain Agnostic)
        prompt = (
            "You are a Linguistic Context Resolver. Convert the 'New Input' into a STANDALONE question.\n"
            "Use the 'Active Context' to resolve ambiguities.\n"
            "\n"
            "--- RULES ---\n"
            "1. ENTITY SWITCH: If input is just a Name/Noun (e.g., 'And the competitor?'), apply the OLD Predicate to the NEW Subject.\n"
            "2. ATTRIBUTE SWITCH: If input asks for a new property (e.g., 'And the cost?'), keep the OLD Subject but change the Predicate.\n"
            "3. NO CONTEXT: If input is a complete question, output it as is.\n"
            "\n"
            f"Active Context: {self.active_context}\n"
            f"New Input: {raw_query}\n\n"
            "Output ONLY the standalone question:"
        )
        
        try:
            rewritten = self.llm.invoke(prompt).content.strip()
            self.console.print(f"[dim]🔎 Logic: '{raw_query}' -> '{rewritten}'[/dim]")
            return rewritten
        except:
            return raw_query

    def _update_state(self, last_a: str, rewritten_q: str):
        """
        Updates the Active Context. 
        Keeps memory clean by summarizing ONLY the current active topic.
        """
        prompt = (
            "You are a Session State Manager. Update the 'Active Context' and 'Current Intent'.\n"
            "1. Active Context: Summarize the CURRENT Subject and Predicate being discussed. Discard finished topics.\n"
            "2. Current Intent: Classify the user's goal (e.g., 'fact retrieval', 'comparison', 'aggregation', 'general chat').\n\n"
            f"Old Context: {self.active_context}\n"
            f"Latest Q: {rewritten_q}\n"
            f"Latest A: {last_a}\n\n"
            "Output JSON format: {\"context\": \"...\", \"intent\": \"...\"}"
        )
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            clean_json = response.replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            
            new_context = data.get("context", "")
            if new_context:
                self.active_context = f"{self.active_context} | {new_context}" if self.active_context else new_context
            self.current_intent = data.get("intent", "General inquiry")
        except:
            self.active_context = f"{self.active_context} | Topic: {rewritten_q}" if self.active_context else f"Topic: {rewritten_q}"

    def _retrieve_long_term_memory(self, query: str) -> str:
        if not self.vs: return ""
        retriever = self.vs.get_hybrid_retriever(k=5)
        if not retriever: return ""
        try:
            docs = retriever.invoke(query)
            # Fetch memory but don't show to user
            memories = [d.page_content for d in docs if d.metadata.get("source_type") == "episodic_memory"]
            return "\n".join(memories) if memories else ""
        except: return ""

    def _calculate_triad_metrics(self, question: str, answer: str, context_chunks: list) -> dict:
        if not context_chunks: return {}
        q_emb = self.vs.emb.embed_query(question)
        a_emb = self.vs.emb.embed_query(answer)
        c_embs = [self.vs.emb.embed_query(c) for c in context_chunks[:3]]
        ctx_rel = np.mean([util.cos_sim(q_emb, c).item() for c in c_embs]) if c_embs else 0
        ans_rel = util.cos_sim(q_emb, a_emb).item()
        return {"context_rel": round(max(0, ctx_rel), 2), "answer_rel": round(max(0, ans_rel), 2)}
    def _handle_math_operations(self, question: str, context_text: str) -> Optional[str]:
        """
        Math Agent: Detects calculation needs on unstructured text/tables.
        """
        # 1. Check if Math is needed (Zero-shot classification)
        check_prompt = (
            f"Question: '{question}'\n"
            "Does answering this require mathematical calculation (sum, average, count, difference) based on the data provided in the context?\n"
            "Answer YES or NO."
        )
        is_math = self.llm.invoke(check_prompt).content.strip().upper()
        
        if "YES" not in is_math:
            return None

        self.console.print("[yellow] Math detected. Executing Code Interpreter...[/yellow]")

        # 2. Generate Python Code
        # We explicitly tell it to parse Markdown tables using io.StringIO
        code_prompt = (
            "You are a Python Data Analyst. The `context_str` variable contains text with data (often in Markdown tables).\n"
            "YOUR GOAL: Write a script to extract the data and answer the User Question.\n"
            "\n"
            "--- GUIDELINES ---\n"
            "1. Use `io.StringIO` and `pd.read_csv(..., sep='|')` to parse Markdown tables if present.\n"
            "2. Clean column names (strip whitespace).\n"
            "3. Perform the calculation (Sum, Mean, etc.).\n"
            "4. PRINT the final descriptive answer.\n"
            "5. Handle errors gracefully (print 'Data not found' if parsing fails).\n"
            "6. Return ONLY executable code. No markdown blocks.\n"
            "\n"
            f"User Question: {question}\n"
            f"context_str = '''{context_text[:6000]}'''\n" # Pass context directly
        )
        
        try:
            code = self.llm.invoke(code_prompt).content.strip()
            code = code.replace("```python", "").replace("```", "").strip()
            
            # Inject necessary imports for the REPL
            full_code = "import pandas as pd\nimport io\nimport numpy as np\n" + code
            
            # 3. Execute
            result = self.python_repl.run(full_code)
            return f"Calculated Result: {result.strip()}"
        except Exception as e:
            return f"Calculation failed: {str(e)}"
    def answer(self, question: str, doc_id: Optional[str] = None):
        t_start = time.time()
        
        # 1. Rewrite Query
        standalone_query = self._rewrite_query(question)
        
        # --- SPECIAL HANDLER: Session Summary ---
                # --- SPECIAL HANDLER: Session Summary ---
        if standalone_query == "__META_SUMMARIZE_SESSION__":
            # Prefer local chat history summary
            sess = self.get_session_summary(turns=20)
            summary = sess.get("summary", "")
            highlights = sess.get("highlights", [])
            # Format output for user
            resp_lines = []
            resp_lines.append(f"**Current Session Context:**\n{self.active_context}\n")
            if summary:
                resp_lines.append(f"**Summary:** {summary}\n")
            if highlights:
                resp_lines.append("**Highlights:**")
                for h in highlights:
                    resp_lines.append(f"- {h}")
            # If no history recorded, show a helpful message
            if not self.chat_history:
                resp_lines.append("No prior chat turns recorded in this session.")
            # Also provide option to show raw recent turns if user asks
            response = "\n".join(resp_lines)
            self.console.print(Panel(response, title="Session Summary", border_style="blue"))
            return


        if standalone_query == "__META_SUMMARIZE_SESSION__":
            # Retrieve internal episodic memory instead of external docs
            history_summary = self._retrieve_long_term_memory("conversation summary")
            response = (
                f"**Current Session Context:**\n{self.active_context}\n\n"
                f"**Detailed History:**\n{history_summary if history_summary else 'No details available yet.'}"
            )
            self.console.print(Panel(response, title="Session Summary", border_style="blue"))
            return

        # 2. Retrieval (External Docs)
        retriever = self.vs.get_hybrid_retriever(k=30) 
        raw_docs = retriever.invoke(standalone_query) if retriever else []
        allowed_docs = [d for d in raw_docs if role_allows(self.role, d.metadata.get("sensitivity", "low"))]
        
        # 3. Context Preparation
        context_text = ""
        citations = []
        chunks_for_metrics = []
        
                # 3. Context Preparation (table-aware citations)
        context_text = ""
        citations = []
        chunks_for_metrics = []

        display_idx = 0  # visible source counter (skips episodic_memory)

        for raw_i, d in enumerate(allowed_docs[:20]):
            # Skip internal memory in user-facing citations
            if d.metadata.get("source_type") == "episodic_memory":
                continue

            display_idx += 1
            src_type = d.metadata.get("source_type", "unknown")
            clean = " ".join(d.page_content.split())
            chunks_for_metrics.append(clean)

            # If this chunk is a table markdown (created from CSV), try to extract the row range and header
            loc_text = ""
            file_label = d.metadata.get("doc_id", "UnknownDoc")
            snippet_header = ""
            if src_type == "table_markdown":
                row_idx = d.metadata.get("row_idx")  # expected format "start-end"
                page_no = d.metadata.get("page", 1)
                if row_idx:
                    loc_text = f"Rows {row_idx} (batch page {page_no})"
                else:
                    loc_text = f"batch page {page_no}"

                # try to parse header line from the chunk (first line starting with '|')
                try:
                    first_lines = clean.splitlines()
                    header_line = None
                    for ln in first_lines[:5]:
                        ln_strip = ln.strip()
                        if ln_strip.startswith("|") and ln_strip.count("|") >= 2:
                            header_line = ln_strip
                            break
                    if header_line:
                        # extract column names and make short label
                        cols = [c.strip() for c in header_line.strip("| ").split("|")]
                        file_label = f"{file_label} [{', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}]"
                        snippet_header = " | ".join(cols[:3])
                    else:
                        snippet_header = clean[:80].replace("\n", " ") + ("..." if len(clean) > 80 else "")
                except Exception:
                    snippet_header = clean[:80].replace("\n", " ") + ("..." if len(clean) > 80 else "")

                # For the LLM context, keep the full chunk but label it clearly
                context_text += f"Source [{display_idx}] (Table rows {row_idx}): {clean}\n\n"

            else:
                # Non-table sources: show page number
                page_no = d.metadata.get("page", 1)
                loc_text = f"Pg {page_no}"
                snippet_header = clean[:80].replace("\n", " ") + ("..." if len(clean) > 80 else "")
                context_text += f"Source [{display_idx}] (Type: {src_type}): {clean}\n\n"

            # Add visual citation (only non-episodic)
            citations.append({
                "id": display_idx,
                "file": d.metadata.get('doc_id'),
                "file_label": file_label,
                "loc": loc_text,
                "snippet": snippet_header
            })


        # 4. MATH LAYER (Pre-computation)
        math_result = self._handle_math_operations(standalone_query, context_text)
        if math_result:
            context_text += f"\n\n>>> SYSTEM CALCULATION RESULT:\n{math_result}\n<<<\n"

        # 5. Generation

        sys_msg = """
        You are a Precision RAG Analyst.

        Answer the question using ONLY the provided Context.

        --- GUIDELINES ---

        1. PRIMARY TRUTH:
        If a "SYSTEM CALCULATION RESULT" is present, it MUST be used as the primary source for any numeric answer.

        2. EXACT DATA:
        If the Context contains explicit, complete, and unambiguous data needed to answer the question,
        provide an exact answer and cite sources.

        3. ESTIMATION MODE (ALLOWED):
        If the Context does NOT contain complete data, but a reasonable numerical ESTIMATE can be derived
        from partial data (e.g., min/max ranges, subsets, or incomplete chapter listings),
        you MAY provide an estimated answer.

        In this case, you MUST:
        - Clearly label the result as "ESTIMATED"
        - Explicitly state the assumption used to derive the estimate
        - Explain briefly what data is missing
        - Use only information present in the Context
        - NEVER invent or assume unseen data

        4. FORBIDDEN:
        - Do NOT hallucinate missing values
        - Do NOT use external knowledge
        - Do NOT silently guess

        5. NO DATA:
        If neither an exact answer nor a reasonable estimate can be derived from the Context,
        respond with:
        "The provided documents do not contain sufficient information to answer this question,
        even approximately."

        6. CONFLICTS:
        If sources conflict, describe the conflict explicitly and do NOT average or reconcile them.

        7. CITATIONS:
        Cite sources using [1], [2], etc., corresponding to the Context.
        """

        
        user_msg = f"CONTEXT:\n{context_text}\n\nQUESTION: {standalone_query}"
        
        try:
            raw_response = self.llm.invoke([("system", sys_msg), ("human", user_msg)]).content
            final_answer = raw_response.strip()
        except Exception as e:
            self.console.print(f"[red]LLM Error: {e}[/red]")
            return
                # append to local chat history (store the user question + final answer)
        try:
            self.chat_history.append({
                "q": standalone_query,
                "a": final_answer,
                "ts": datetime.now().isoformat()
            })
            # trim to max size
            if len(self.chat_history) > self.max_history_turns:
                self.chat_history = self.chat_history[-self.max_history_turns:]
            # persist the session (includes chat_history)
            self._save_session()
        except Exception:
            pass

        # 6. Update State & Save
        self._update_state(final_answer, standalone_query)
        self._save_session()
        self.vs.add_memory_trace(f"Q: {standalone_query}\nA: {final_answer}", self.role)

        # 7. Output
        metrics = self._calculate_triad_metrics(standalone_query, final_answer, chunks_for_metrics)
        
        self.console.print(Panel(final_answer, title=f"DH-RAG (Intent: {self.current_intent})", border_style="green"))
        
        if citations:
            # Deduplicate visual citations
            seen = set()
            unique_citations = []
            for c in citations:
                if c['id'] not in seen:
                    unique_citations.append(c)
                    seen.add(c['id'])
            
            ref_table = Table(title="External Sources", box=box.SIMPLE)
            ref_table.add_column("Ref"); ref_table.add_column("Doc"); ref_table.add_column("Loc")
            for c in unique_citations[:10]:
                # Use file_label if available for more readable doc name
                doc_display = c.get("file_label") or c.get("file")
                loc_display = c.get("loc") or ""
                # Show a short snippet inline as well for quick scanning
                snippet = c.get("snippet", "")
                ref_table.add_row(f"[{c['id']}]", doc_display, f"{loc_display}\n{snippet}")


            self.console.print(ref_table)

        self.console.print(f"[dim]Metrics: CtxRel={metrics.get('context_rel',0)} | AnsRel={metrics.get('answer_rel',0)}[/dim]")