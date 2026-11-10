"""Business service layer exports."""

from typing import Any, Dict, List, Tuple

from qna_bot import resolve_lazy_export

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AIAgent": ("qna_bot.services.agent_service", "AIAgent"),
    "ApprovalService": ("qna_bot.services.approval_service", "ApprovalService"),
    "AuthService": ("qna_bot.services.auth_service", "AuthService"),
    "ChatService": ("qna_bot.services.chat_service", "ChatService"),
    "ChatTurnResult": ("qna_bot.services.chat_service", "ChatTurnResult"),
    "ConversationService": ("qna_bot.services.conversation_service", "ConversationService"),
    "DocumentService": ("qna_bot.services.document_service", "DocumentService"),
    "MCPService": ("qna_bot.services.mcp_service", "MCPService"),
    "SuggestedPrompt": ("qna_bot.services.suggestion_service", "SuggestedPrompt"),
    "SuggestionService": ("qna_bot.services.suggestion_service", "SuggestionService"),
    "UserService": ("qna_bot.services.user_service", "UserService"),
    "build_langchain_messages": ("qna_bot.services.agent_service", "build_langchain_messages"),
    "clear_current_chat": ("qna_bot.services.session_service", "clear_current_chat"),
    "export_as_json": ("qna_bot.services.export_service", "export_as_json"),
    "export_as_markdown": ("qna_bot.services.export_service", "export_as_markdown"),
    "export_as_text": ("qna_bot.services.export_service", "export_as_text"),
    "initialize_session_state": ("qna_bot.services.session_service", "initialize_session_state"),
    "restore_chat": ("qna_bot.services.session_service", "restore_chat"),
    "save_current_chat_to_recent": ("qna_bot.services.session_service", "save_current_chat_to_recent"),
    "start_new_chat": ("qna_bot.services.session_service", "start_new_chat"),
    "stream_chunks": ("qna_bot.services.agent_service", "stream_chunks"),
    "validate_and_encode_image": ("qna_bot.services.image_service", "validate_and_encode_image"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, globals(), _EXPORTS, name)


def __dir__() -> List[str]:
    return sorted(__all__)
