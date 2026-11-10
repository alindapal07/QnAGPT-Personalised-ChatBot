"""Discovery and retrieval of MCP server prompts."""

import threading
from typing import Any, Dict, List, Optional, Sequence

from qna_bot.mcp.models import DiscoveredPrompt
from qna_bot.mcp.security import sanitize_tool_output


class MCPPromptManager:
    """Stores discovered MCP prompt templates per user/server."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._prompts_by_user: Dict[str, Dict[tuple[str, str], DiscoveredPrompt]] = {}

    def register_server_prompts(
        self,
        user_id: str,
        server_id: str,
        server_name: str,
        raw_prompts: Sequence[object],
    ) -> List[DiscoveredPrompt]:
        with self._lock:
            user_map = self._prompts_by_user.setdefault(user_id, {})
            for key in [k for k in user_map if k[0] == server_id]:
                user_map.pop(key, None)

            items: List[DiscoveredPrompt] = []
            for raw in raw_prompts:
                name = str(getattr(raw, "name", ""))
                if not name:
                    continue
                args = []
                for arg in getattr(raw, "arguments", None) or []:
                    args.append(
                        {
                            "name": str(getattr(arg, "name", "")),
                            "description": str(getattr(arg, "description", "") or ""),
                            "required": bool(getattr(arg, "required", False)),
                        }
                    )
                item = DiscoveredPrompt(
                    name=name,
                    server_id=server_id,
                    server_name=server_name,
                    user_id=user_id,
                    description=str(getattr(raw, "description", "") or ""),
                    arguments=args,
                )
                user_map[(server_id, name)] = item
                items.append(item)
            return items

    def clear_server_prompts(self, user_id: str, server_id: str) -> None:
        with self._lock:
            user_map = self._prompts_by_user.get(user_id, {})
            for key in [k for k in user_map if k[0] == server_id]:
                user_map.pop(key, None)

    def list_prompts(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredPrompt]:
        with self._lock:
            items = list(self._prompts_by_user.get(user_id, {}).values())
            if server_id:
                items = [p for p in items if p.server_id == server_id]
            return sorted(items, key=lambda p: (p.server_name, p.name))

    @staticmethod
    def format_prompt_result(raw_result: Any) -> Dict[str, Any]:
        messages = getattr(raw_result, "messages", None) or []
        formatted = []
        for msg in messages:
            role = str(getattr(msg, "role", "user"))
            content_obj = getattr(msg, "content", "")
            text = getattr(content_obj, "text", None)
            formatted.append(
                {
                    "role": role,
                    "content": sanitize_tool_output(str(text if text is not None else content_obj)),
                }
            )
        return {
            "description": str(getattr(raw_result, "description", "") or ""),
            "messages": formatted,
        }
