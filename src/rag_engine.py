import os
import time
import re
from typing import Optional, List, Dict, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI

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
        
        main_model_name = "gemini-2.5-flash" if model_type == "pro" else "gemini-2.5-flash"
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY env var missing")

        self.llm = ChatGoogleGenerativeAI(
            model=main_model_name,
            temperature=0,
        )
        
        # LLM-as-a-Judge
        self.judge_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0
        )

    def answer(self, question: str, doc_id: Optional[str] = None):
        t_start = time.time()
        metrics = {
            "retrieval_ms": 0, "llm_ms": 0, "eval_ms": 0,
            "candidates_fetched": 0, "chunks_used": 0, "filtered_out": 0,
            "faithfulness": 0.0, "relevance": 0.0 
        }

        # 1. Broad Retrieval 
        retriever = self.vs.get_hybrid_retriever(k=60)
        if not retriever:
             self.console.print("[red]Database is empty.[/red]")
             return

        t0 = time.time()
        raw_candidates = retriever.invoke(question)
        metrics["retrieval_ms"] = round((time.time() - t0) * 1000, 2)
        metrics["candidates_fetched"] = len(raw_candidates)

        # 2. Security Filtering
        valid_docs = []
        for doc in raw_candidates:
            meta = doc.metadata
            if doc_id and meta.get("doc_id") != doc_id:
                metrics["filtered_out"] += 1
                continue
            if not role_allows(self.role, meta.get("sensitivity", "low")):
                metrics["filtered_out"] += 1
                continue
            valid_docs.append(doc)

        final_context = valid_docs[:15]
        metrics["chunks_used"] = len(final_context)
        
        if not final_context:
            self._print_response("Information not found (Access Denied or No Data).", [], metrics, t_start)
            return

        # 3. Context Preparation
        context_parts = []
        citations_data = []
        full_context_text = "" # For judge
        
        for i, d in enumerate(final_context):
            meta = d.metadata
            ref_idx = i + 1
            
            if meta.get("source_type") == "table_markdown":
                loc = f"Row: {meta.get('row_idx')}"
                icon = "📊"
            else:
                loc = f"Page: {meta.get('page')}"
                icon = "📄"
            
            citations_data.append({
                "id": ref_idx, "file": meta.get('doc_id'), 
                "loc": loc, "type": icon, "access": meta.get("sensitivity", "low")
            })
            
            clean_text = " ".join(d.page_content.split())
            context_parts.append(f"Source [{ref_idx}]: {clean_text}")
            full_context_text += f"\nSource [{ref_idx}]: {clean_text}"

        # 4. Generation
        prompt_text = self._build_prompt(question, context_parts)
        t1 = time.time()
        try:

            raw_response = self.llm.invoke(prompt_text).content
            response_text = self._normalize_llm_text(raw_response)
        except Exception as e:
            self._print_response(
                "⚠️ Temporary LLM error. Please retry later.",
                [],
                metrics,
                t_start
            )
            return

        metrics["llm_ms"] = round((time.time() - t1) * 1000, 2)

        # 5. Quality Evaluation (LLM-as-a-Judge)
        t2 = time.time()
        f_score, r_score = self._evaluate_quality(question, response_text, full_context_text)
        metrics["faithfulness"] = f_score
        metrics["relevance"] = r_score
        metrics["eval_ms"] = round((time.time() - t2) * 1000, 2)
        
        # 6. Output
        self._print_response(response_text, citations_data, metrics, t_start)

    def _evaluate_quality(self, question: str, answer: str, context: str) -> Tuple[float, float]:
        faith_prompt = (
            "You are a strict evaluator. Analyze if the ANSWER is strictly supported by the CONTEXT.\n"
            "Return a single numeric score between 0.0 and 1.0 (inclusive) on the first line, "
            "optionally followed by a short explanation on subsequent lines.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER: {answer}\n\n"
            "Return ONLY the number on the first line, e.g. '0.85'."
        )

        rel_prompt = (
            "You are a strict evaluator. Analyze if the ANSWER directly addresses the QUESTION.\n"
            "Return a single numeric score between 0.0 and 1.0 (inclusive) on the first line, "
            "optionally followed by a short explanation.\n\n"
            f"QUESTION: {question}\n"
            f"ANSWER: {answer}\n\n"
            "Return ONLY the number on the first line, e.g. '0.95'."
        )

        def parse_score(text: str) -> float:
            if not text:
                return 0.0
            t = text.strip()
            # Try common patterns: 0.85 or 1.0 or 0 or 1
            m = re.search(r"(?<!\d)(0(?:[.,]\d+)?|1(?:[.,]0+)?)(?!\d)", t)
            if m:
                s = m.group(1).replace(",", ".")
                try:
                    val = float(s)
                    return max(0.0, min(1.0, val))
                except Exception:
                    pass
            # Try percent like "85%" or "85.0 %"
            m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*%", t)
            if m2:
                try:
                    val = float(m2.group(1).replace(",", "."))
                    return max(0.0, min(1.0, val / 100.0))
                except Exception:
                    pass
            # Try bare number possibly >1 (e.g. "85" meaning 85%)
            m3 = re.search(r"(?<!\d)(\d+(?:[.,]\d+)?)(?!\d)", t)
            if m3:
                try:
                    val = float(m3.group(1).replace(",", "."))
                    if val > 1.0:
                        # interpret as percentage
                        val = val / 100.0
                    return max(0.0, min(1.0, val))
                except Exception:
                    pass
            return 0.0

        try:
            f_resp = self.judge_llm.invoke(faith_prompt).content
            r_resp = self.judge_llm.invoke(rel_prompt).content


            self.console.print(f"[dim]Judge faith raw response:[/dim]\n{f_resp}\n")
            self.console.print(f"[dim]Judge rel  raw response:[/dim]\n{r_resp}\n")

            f_score = parse_score(f_resp)
            r_score = parse_score(r_resp)
            return f_score, r_score
        except Exception as e:
            self.console.print(f"[red]Judge invocation error:[/red] {e}")
            return 0.0, 0.0


    def _build_prompt(self, question, context):
        ctx = "\n\n".join(context)
        sys = (
            "You are a secure corporate analyst. Use ONLY the provided CONTEXT. "
            "If the answer is missing, state 'Information not found'. "
            "Use [1], [2] citation style."
        )
        return [("system", sys), ("human", f"CONTEXT:\n{ctx}\n\nQUESTION: {question}")]

    def _print_response(self, text: str, citations: List[Dict], metrics: Dict, t_start: float):
        total_time = round(time.time() - t_start, 2)
        
        # Answer
        self.console.print(Panel(text, title="[bold green]RAG Response[/bold green]", border_style="green"))
        
        # Sources
        if citations:
            ref_table = Table(title="Sources Used", box=box.SIMPLE)
            ref_table.add_column("Ref", justify="center", style="cyan")
            ref_table.add_column("Type", justify="center")
            ref_table.add_column("Document", style="magenta")
            ref_table.add_column("Location", style="yellow")
            
            for c in citations:
                ref_table.add_row(f"[{c['id']}]", c['type'], c['file'], c['loc'])
            self.console.print(ref_table)

        # Quality Metrics 
        def color_score(score):
            if score >= 0.8: return f"[green]{score}[/green]"
            if score >= 0.5: return f"[yellow]{score}[/yellow]"
            return f"[red]{score}[/red]"

        q_table = Table(box=box.ROUNDED, show_header=True, title="Quality Assurance (LLM-as-a-Judge)")
        q_table.add_column("Metric")
        q_table.add_column("Score (0-1)")
        q_table.add_column("Meaning")
        
        q_table.add_row(
            "Faithfulness", 
            color_score(metrics['faithfulness']), 
            "Is the answer grounded in context?"
        )
        q_table.add_row(
            "Relevance", 
            color_score(metrics['relevance']), 
            "Does it answer the user's question?"
        )
        self.console.print(q_table)

        # Technical Metrics
        tech_table = Table(box=box.SIMPLE, show_header=False)
        tech_table.add_row(f"⏱️ Total: {total_time}s", f"Pipe: {metrics['candidates_fetched']}->{metrics['chunks_used']} docs")
        self.console.print(tech_table)
        
    def _normalize_llm_text(self, content) -> str:

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # Gemini structured response: list[dict]
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        texts.append(item["text"])
            return "\n".join(texts).strip()

        # dict fallback
        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                return content["text"]

        # last resort
        return str(content)
