# QnAGPT Service Contracts & API Reference

**Project**: QnAGPT – Production-Ready Multi-User Agentic AI Workspace  
**Status**: Verified Service APIs  
**Author**: Principal Software Architect & Senior Python Backend Engineer  

---

## 1. Web Endpoints & Callback Interface

As a Streamlit-driven application, the primary entry point is an HTTP SPA server:

### Inbound Web Routes
- **`GET /`**: Main application interface. Boots session state, checks bearer tokens, and renders the active workspace view or Auth Gate.
- **`GET /?code={code}&state={state}`**: OAuth2 Callback handler. Handled automatically during `app.py` bootstrapping via `handle_oauth_callback_params()`:
  - Validates `state` against session-stored CSRF nonce.
  - Exchanges `code` for OAuth access tokens using PKCE verifier.
  - Sets authenticated bearer session token and rewrites browser URL.

---

## 2. Core Service Contracts (`qna_bot/services/`)

All high-level domain operations are encapsulated in strongly-typed service classes.

### 2.1. `AuthService` (`qna_bot/services/auth_service.py`)

```python
class AuthService:
    def register(self, payload: RegisterRequest) -> UserSession:
        """Register a new user with email and password.
        Raises: ValidationError, ConflictError, RateLimitExceededError."""

    def login(self, payload: LoginRequest) -> UserSession:
        """Authenticate with email and password.
        Raises: AuthenticationError, RateLimitExceededError."""

    def validate_session(self, session_token: Optional[str]) -> Optional[UserSession]:
        """Validate an opaque bearer token against active user_sessions."""

    def logout(self, session_token: Optional[str]) -> None:
        """Explicitly revoke an active session token."""

    def initiate_oauth_flow(self, provider: str) -> str:
        """Generate PKCE challenge, save CSRF state, and return authorization URL."""

    def complete_oauth_flow(self, provider: str, code: str, state: str) -> UserSession:
        """Complete authorization code exchange and issue user session."""
```

---

### 2.2. `ChatService` (`qna_bot/services/chat_service.py`)

```python
class ChatService:
    def stream_response(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        chat_mode: str = "Auto (Agentic)",
        system_prompt: str = "",
        model_name: Optional[str] = None,
        images: Sequence[Any] = (),
    ) -> Generator[str, None, ChatTurnResult]:
        """Execute a stateful turn through LangGraph, yielding tokens for streaming.
        Returns ChatTurnResult with final text, citations, tool executions, and latency."""

    def clear_conversation(self, user_id: str, conversation_id: str) -> None:
        """Clear all messages from an active conversation thread."""
```

---

### 2.3. `DocumentService` (`qna_bot/services/document_service.py`)

```python
class DocumentService:
    def ingest_files(
        self,
        files: Sequence[Tuple[str, bytes]],
        user_id: str,
    ) -> List[IngestionResult]:
        """Validate, store locally, extract text, chunk, and index documents into Qdrant."""

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Query all stored documents with processing status and chunk counts."""

    def remove_document(self, document_id: str, user_id: str) -> bool:
        """Delete document original file, vector chunks, and database records."""

    def retry_document(self, document_id: str, user_id: str) -> IngestionResult:
        """Re-index a previously stored document that encountered indexing errors."""

    def vector_health(self) -> Dict[str, Any]:
        """Check live connectivity and collection status of Qdrant vector engine."""
```

---

### 2.4. `MCPService` (`qna_bot/services/mcp_service.py`)

```python
class MCPService:
    def register_server(
        self,
        user_id: str,
        name: str,
        transport_type: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> MCPServerRecord:
        """Register a new Stdio or SSE MCP server with SSRF and command validation."""

    def list_tools(
        self,
        user_id: str,
        include_disabled: bool = False,
    ) -> List[DiscoveredTool]:
        """Discover tools across all connected MCP servers and built-in suites."""

    def set_tool_permission(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        category: str,
        auto_approve: bool = False,
        enabled: bool = True,
    ) -> None:
        """Update tool governance policy (READ_ONLY, MODIFICATION, SENSITIVE, BLOCKED)."""

    def list_audit_logs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Query execution logs with masked arguments and latency benchmarks."""
```

---

### 2.5. `ApprovalService` (`qna_bot/services/approval_service.py`)

```python
class ApprovalService:
    def create_pending_approval(
        self,
        user_id: str,
        thread_id: str,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        category: str,
        reason: str,
    ) -> ApprovalRecord:
        """Create a pending approval request with cryptographic arguments hash."""

    def verify_and_consume(
        self,
        approval_id: str,
        user_id: str,
        thread_id: str,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """Verify that human approval was explicitly granted for the exact arguments."""
```

---

### 2.6. `ExportService` (`qna_bot/services/export_service.py`)

```python
def export_as_json(conversation_id: str, messages: Sequence[BaseMessage]) -> str:
    """Serialize conversation into valid RFC 8259 JSON."""

def export_as_markdown(messages: Sequence[BaseMessage]) -> str:
    """Serialize conversation into clean Markdown with blockquotes and citations."""

def export_as_text(messages: Sequence[BaseMessage]) -> str:
    """Serialize conversation into plain UTF-8 text."""
```

---

## 3. Outbound External API Contracts

| Target Service | Protocol | Endpoint Pattern | Purpose |
| :--- | :--- | :--- | :--- |
| **Groq Cloud API** | HTTPS / REST | `https://api.groq.com/openai/v1/chat/completions` | High-speed LLM text completion and tool calling |
| **Qdrant Vector DB** | HTTP / REST | `http://localhost:6333/collections/{collection}/points` | Dense vector similarity search with user payload filters |
| **DuckDuckGo** | HTTPS / HTML | Public web search endpoints via `ddgs` | Live web discovery without API key requirements |
| **Official MCP Registry** | HTTPS / REST | `https://registry.modelcontextprotocol.io/v0/servers` | Official Model Context Protocol directory lookup |
| **Glama AI Registry** | HTTPS / REST | `https://glama.ai/api/mcp/v1/servers` | Extended MCP community registry search |
| **Target Webpages** | HTTPS / HTTP | Dynamic user-specified URLs | SSRF-safe page fetching and Markdown table extraction |
