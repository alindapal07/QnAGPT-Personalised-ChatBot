"""Discovery and authorized reading of MCP server resources."""

import threading
from typing import Any, Dict, List, Optional, Sequence

from qna_bot.mcp.config import MAX_RESOURCE_BYTES
from qna_bot.mcp.models import DiscoveredResource
from qna_bot.mcp.security import sanitize_tool_output


class MCPResourceManager:
    """Stores discovered MCP resources per user/server and safely formats read payloads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources_by_user: Dict[str, Dict[tuple[str, str], DiscoveredResource]] = {}

    def register_server_resources(
        self,
        user_id: str,
        server_id: str,
        server_name: str,
        raw_resources: Sequence[object],
    ) -> List[DiscoveredResource]:
        with self._lock:
            user_map = self._resources_by_user.setdefault(user_id, {})
            for key in [k for k in user_map if k[0] == server_id]:
                user_map.pop(key, None)

            items: List[DiscoveredResource] = []
            for raw in raw_resources:
                uri = str(getattr(raw, "uri", ""))
                if not uri:
                    continue
                mime = str(
                    getattr(raw, "mime_type", None)
                    or getattr(raw, "mimeType", None)
                    or "text/plain"
                )
                item = DiscoveredResource(
                    uri=uri,
                    name=str(getattr(raw, "name", "") or uri),
                    server_id=server_id,
                    server_name=server_name,
                    user_id=user_id,
                    description=str(getattr(raw, "description", "") or ""),
                    mime_type=mime,
                )
                user_map[(server_id, uri)] = item
                items.append(item)
            return items

    def clear_server_resources(self, user_id: str, server_id: str) -> None:
        with self._lock:
            user_map = self._resources_by_user.get(user_id, {})
            for key in [k for k in user_map if k[0] == server_id]:
                user_map.pop(key, None)

    def list_resources(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredResource]:
        with self._lock:
            items = list(self._resources_by_user.get(user_id, {}).values())
            if server_id:
                items = [r for r in items if r.server_id == server_id]
            return sorted(items, key=lambda r: (r.server_name, r.uri))

    @staticmethod
    def format_read_result(raw_result: Any) -> Dict[str, str]:
        contents = getattr(raw_result, "contents", None) or []
        parts: List[str] = []
        mime_type = "text/plain"
        for block in contents:
            block_mime = getattr(block, "mime_type", None) or getattr(block, "mimeType", None)
            if block_mime:
                mime_type = str(block_mime)
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
            else:
                blob = getattr(block, "blob", None)
                if blob is not None:
                    parts.append(f"[binary resource payload: {len(str(blob))} base64 chars]")

        joined = sanitize_tool_output("\n\n".join(parts), max_chars=MAX_RESOURCE_BYTES)
        return {"mime_type": mime_type, "content": joined}
