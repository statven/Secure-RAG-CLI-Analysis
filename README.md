##Secure Context-Aware CLI RAG for Document Analysis
A command-line tool that uses Retrieval-Augmented Generation (RAG) to analyze user-provided documents (PDF, DOCX, TXT, CSV, XLSX) with strict Role-Based Access Control (RBAC), precise citations, and provable, evidence-backed answers.

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Key Features](#key-features)
3. [Architecture Overview](#architecture-overview)
4. [Getting Started](#getting-started)

   * [Requirements](#requirements)
   * [Install](#install)
   * [Environment Variables (.env.template)](#environment-variables-envtemplate)
5. [Usage](#usage)

   * [Single-shot Query Example](#single-shot-query-example)
   * [Interactive REPL Example](#interactive-repl-example)
6. [CLI Reference](#cli-reference)
7. [Ingestion & Data Model](#ingestion--data-model)

   * [Supported Formats](#supported-formats)
   * [Chunk / Metadata Schema](#chunk--metadata-schema)
   * [Sensitivity Tagging](#sensitivity-tagging)
8. [RAG Engine, Prompting & Citations](#rag-engine-prompting--citations)

   * [Prompt Template (recommended)](#prompt-template-recommended)
   * [Citation Format](#citation-format)
   * [Hallucination Guardrails](#hallucination-guardrails)
9. [Security & RBAC](#security--rbac)

   * [Role Examples](#role-examples)
   * [Policy Example (YAML)](#policy-example-yaml)
   * [Audit & Operational Notes](#audit--operational-notes)
10. [Evaluation Criteria & Tests](#evaluation-criteria--tests)
11. [Timeline & Milestones (High Level)](#timeline--milestones-high-level)
12. [Development Guidelines](#development-guidelines)
13. [Deliverables](#deliverables)
14. [Contributing & Code of Conduct](#contributing--code-of-conduct)


---

# Project Summary

This project aims to provide a trustworthy, auditable CLI for querying documents. Answers must be strictly grounded in the ingested documents, with every assertion accompanied by a citation (page/row/chunk). Sensitive passages are tagged during ingestion and must be filtered or redacted according to the invoking user role.

---

# Key Features

* Multi-format ingestion: PDF, DOCX, TXT, CSV, XLSX.
* Local vector store (FAISS) with metadata support for precise retrieval.
* RBAC: role flag (`--role`) controls what content is visible; sensitive content is redacted or denied.
* Citation-first answers: every factual claim is accompanied by an exact source reference.
* CLI (Typer) with interactive and single-shot modes.
* Pluggable LLM / embeddings and provider abstraction layer.
* Test suite and CI skeleton for reproducible quality.

---

# Architecture Overview

High-level components:

* **CLI (Typer)** - user interface, handles args, role, and I/O formatting (Rich).
* **Ingestion** - file type detection, parsing, normalization, chunking, and sensitivity tagging.
* **Embeddings** - converts chunks to dense vectors via pluggable provider (HF models).
* **Vector Store** - FAISS wrapper storing vectors + metadata (doc_id, page, chunk_id, sensitivity...).
* **Retrieval** - similarity search + metadata filters (role enforcement applied pre-or post-retrieval).
* **RAG Engine** - compose prompt, call LLM, format answers + citations, hallucination checks.
* **Security Layer** - RBAC policies, redaction rules, audit logging.

Diagram (concept):

```
CLI -> Ingestion -> Chunking/Tagging -> Embeddings -> FAISS
             ^                                      |
             |                                      v
           Files                               Retrieval -> RAG Engine -> CLI
                                                 |
                                            Security Filter
```

---

# Getting Started

## Requirements

* Python 3.10+
* Recommended: Linux/macOS for development (Windows supported)
* Optional: Docker for containerization

## Install

```bash
# install packages
pip install -r requirements.txt
```

> The `requirements.txt` includes runtime packages like Typer, Rich, LangChain, FAISS (faiss-cpu), pandas, PyMuPDF, python-docx, sentence-transformers etc. Pin versions for reproducible installs before release.

## Environment Variables (.env.template)

Create `.env` from `.env.template`. Typical vars:

```
# .env.template
API_KEY=        
LLM_PROVIDER=local     # example values: local|openai|google
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_DIR=./indices
AUDIT_LOG=./logs/audit.log
```

**Do not commit** secrets to VCS.

---

# Usage

## Single-shot Query Example

```bash
python rag.py query \
  --file ./data/contracts/example.pdf \
  --role low_rank \
  "What is the project timeline?"
```

Expected outputs:

* If answer found and role allows access:

  ```
  Answer: The timeline is 6 weeks. (See: example.pdf - Page 2, Chunk 3)
  ```
* If found but restricted:

  ```
  Access Denied: The requested information is restricted for role 'low_rank'.
  ```
* If not found in the document:

  ```
  Information not found.
  ```

## Interactive REPL Example

```bash
python rag.py interactive --file ./data/contracts/contract.docx --role high_rank
# then enter natural language queries repeatedly
```

---

# CLI Reference

Main commands (examples; see `--help` for full):

```
python rag.py query --file <path> --role <low_rank|high_rank|admin> "<question>"
python rag.py interactive --file <path> --role <role>
python rag.py ingest --file <path> [--index-path <dir>]   # create / update index
python rag.py validate-index --index-path <dir>
```

Flags:

* `--role` (required): the invoking user's role (RBAC enforced).
* `--index-path`: location to persist FAISS indices and metadata.
* `--provider`: optional LLM/embedding provider override.

---

# Ingestion & Data Model

## Supported Formats

* Unstructured: `.pdf` (PyMuPDF/PyPDF2), `.docx` (python-docx), `.txt`.
* Tabular: `.csv`, `.xlsx` (pandas / openpyxl). Tabular rows should be converted to sentence chunks with precise row/column metadata for citation.

## Chunk / Metadata Schema

Each chunk stored in the vector store SHOULD include:

```json
{
  "doc_id": "example.pdf",
  "page": 2,                 // for page-based sources
  "row": null,               // for tabular sources
  "column": null,
  "chunk_id": "example.pdf::p2::c3",
  "start_char": 123,
  "end_char": 456,
  "text": "Extracted text...",
  "sensitivity": "public",   // public|internal|secret
  "source_path": "./data/..."
}
```

This schema enables deterministic citations: `See: {doc_id} - Page {page}, Chunk {chunk_id}` or `See: {doc_id} - Row {row}, Column '{column}'`.

## Sensitivity Tagging

* Tagging should happen at ingestion using deterministic rules:

  * Metadata patterns (e.g., column name `salary`, `ssn`) => `secret`.
  * Keyword lists and regex (e.g., "trade secret", "confidential") => `internal` or `secret`.
  * Manual overrides via an ingestion config file are supported.

---

# RAG Engine, Prompting & Citations

## Prompt Template (recommended)

**System prompt** (force evidence use):

```
You are an evidence-first assistant. Answer using *only* the provided document chunks. 
Every factual statement must be followed by an explicit citation of the chunk used.
If information is not found, reply exactly: "Information not found."
If the user's role prevents access to supporting chunks, reply: "Access Denied."
Format:
Answer: <concise answer>
Citations:
- <doc_id> - Page <p>, Chunk <id>
- <doc_id2> - Row <r>, Column '<col>'
```

**User prompt**: the question + retrieved chunks appended below.

## Citation Format

* For pages: `See: {doc_id} - Page {page}, Chunk {chunk_id}`
* For tabular: `See: {doc_id} - Row {row}, Column '{column}'`
* The system should print both the answer and a bullet list of citations.

## Hallucination Guardrails

* If the LLM produces claims beyond retrieved chunks, the orchestrator must:

  * Re-check each claim against the top-k retrieved chunks.
  * If claims lack evidence, return `Information not found` or trim claims.
* Use generation temperature 0 (or deterministic LLMs) and explicit instruction to quote sources.

---

# Security & RBAC

## Role Examples

* `low_rank` - minimal access; no `internal`/`secret` info.
* `high_rank` - broader access; can see `internal` but not `secret`.
* `admin` - full access.

## Policy Example (YAML)

```yaml
roles:
  low_rank:
    allow: ["public"]
  high_rank:
    allow: ["public", "internal"]
  admin:
    allow: ["public", "internal", "secret"]

sensitivity_rules:
  - name: salary_column
    pattern: "(?i)salary|compensation"
    assign: secret
  - name: confidential_keyword
    pattern: "(?i)confidential|trade secret"
    assign: internal
```

Policies are checked during retrieval; chunks not allowed for role are filtered out.

## Audit & Operational Notes

* Log every query: user_role, query_text, timestamp, retrieved_chunk_ids (hashed for privacy), redaction events.
* Store audit logs in append-only files or a secure log service.
* Never store raw secrets in logs; hash or redact sensitive fields.

---

# Evaluation Criteria & Tests

Automated tests should cover:

* **Faithfulness**: responses strictly tied to document evidence.
* **Format versatility**: ingestion + query for PDF, DOCX, TXT, CSV, XLSX.
* **Citation accuracy**: citations point to right chunk/page/row.
* **Security compliance**: low_rank cannot access secret content and receives `Access Denied`.
* **Usability**: CLI flags and help are clear.
* **Code quality**: PEP8, docstrings, unit/integration tests.

Create test fixtures in `/data` (multi-column PDF, long DOCX, tabular CSV/XLSX) and tests in `/tests`.

---

# Timeline & Milestones (High Level)

* **Week 1**: Research & architecture; finalize models and CLI args.
* **Week 2**: Ingestion pipeline and chunking.
* **Week 3**: Vector store integration + embeddings.
* **Week 4**: RAG prompts, LLM integration, citation enforcement.
* **Week 5**: CLI integration and RBAC enforcement.
* **Week 6**: Testing, security validation, docs and demo.

---

# Development Guidelines

## Coding & Style

* Follow **PEP8**. Use black / flake8 in pre-commit hooks.
* Type hints for public functions. Docstrings for modules/functions.

## Recommended Branching

* `main` - release branch (protected).
* `develop` - integration branch.
* `feature/*`, `hotfix/*` per task.
* PRs require 1 reviewer + CI pass.

## CI / Pre-commit

* CI steps: lint → unit tests → building index smoke test (optional).
* Pre-commit config: black, isort, flake8, end-of-file fixer.

---

# Deliverables

* GitHub repository with full source and docs.
* `requirements.txt` and `.env.template`.
* Main CLI script `rag.py` and package under `src/`.
* Unit & integration tests in `tests/`.
* Demo GIF or short video demonstrating secure queries and citation output.
* `docs/` with architecture, onboarding and operational guidance.

---

# Contributing & Code of Conduct

* See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for contribution rules and etiquette.
* Use Conventional Commits: `feat:`, `fix:`, `chore:`, etc. Include issue reference when applicable.

---
