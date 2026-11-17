# QnAGPT Developer Setup & Contribution Guide

**Project**: QnAGPT – Production-Ready Multi-User Agentic AI Workspace  
**Status**: Developer Standards & Workflows  
**Author**: Principal Software Architect & Senior Python Backend Engineer  

---

## 1. Prerequisites

- **Python**: Version `3.12+` (verified on Python 3.12.8).
- **Package Manager**: **`uv`** (strongly recommended for sub-second dependency resolution) or `pip`.
- **Docker**: Optional, for running local Qdrant vector database container.
- **API Keys**: A free **Groq Cloud API key** from [console.groq.com](https://console.groq.com).

---

## 2. Local Environment Setup

### 1. Clone & Initialize Environment
```powershell
git clone https://github.com/alindapal07/QnAGPT-Personalised-ChatBot.git
cd langchain_01

# Using uv (fastest)
uv sync

# Or using standard python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and set your Groq API key:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Start Local Vector Database (Qdrant)
If running Qdrant locally in Docker:
```powershell
docker run -d --name qnagpt-qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v "${PWD}\data\qdrant_storage:/qdrant/storage" `
  qdrant/qdrant:latest
```
*(Alternatively, configure `QDRANT_MODE=cloud` with your Qdrant Cloud URL and API key in `.env`)*.

### 4. Apply Database Migrations
Initialize your SQLite relational platform schema:
```powershell
uv run alembic upgrade head
```

---

## 3. Running the Application

### Via Streamlit Runner
```powershell
uv run streamlit run qna_bot/app.py
```

### Via CLI Console Script
The package installs a console binary `qna-bot` via `pyproject.toml`:
```powershell
uv run qna-bot
```

Open your browser at `http://localhost:8501`.

---

## 4. Running the Test Suite

The repository contains 110 automated tests covering unit, integration, and offline UI simulations.

### Run All Tests
```powershell
uv run pytest -v -k "not test_app_renders_all_workspace_pages_cleanly"
```

### Run Specific Subsystems
```powershell
# Corrective RAG (CRAG) Suite (27 tests)
uv run pytest tests/test_crag.py -v

# LangGraph Agent & State Machine Suite (7 tests)
uv run pytest tests/test_agent.py -v

# MCP Client & Security Suite (12 tests)
uv run pytest tests/test_mcp_client.py tests/test_mcp_security.py tests/test_mcp_tools.py -v

# Platform Foundations & Authentication Suite
uv run pytest tests/test_platform_foundation.py tests/test_session_service.py -v
```

---

## 5. Architectural Coding Standards

When contributing code to QnAGPT, adhere strictly to the following architectural rules:

1. **No UI Leakage**: Never import `streamlit` inside `qna_bot/services/`, `qna_bot/agent/`, `qna_bot/rag/`, or `qna_bot/repositories/`. Streamlit is restricted strictly to `qna_bot/ui/` and `qna_bot/app.py`.
2. **Mandatory Tenant Scoping**: Every database query, vector search payload filter, and storage directory path **must** accept and enforce `user_id`. Never execute unscoped database reads or deletions.
3. **Decoupled LLM Tooling**:
   - Use `_make_agent_llm()` with `tool_choice="auto"` for function selection.
   - Use `_make_response_llm()` without tools for final text synthesis to prevent model tool-call formatting errors.
4. **Defensive Error Handling**: Catch and map low-level library exceptions into domain exceptions (`AppError`, `NotFoundError`, `ConflictError`, `ValidationError`) defined in `qna_bot/core/exceptions.py`.
5. **No Secrets in Logs or UI**: Sensitive arguments must always pass through `mask_secrets()` or `format_masked_arguments()` before being logged or displayed.
