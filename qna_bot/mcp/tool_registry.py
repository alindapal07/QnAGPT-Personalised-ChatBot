"""User-scoped registry of dynamically discovered MCP tools."""

import re
import threading
import time
from typing import Dict, List, Optional, Sequence

from qna_bot.database.repositories import PermissionRepository
from qna_bot.mcp.models import DiscoveredTool
from qna_bot.mcp.permissions import infer_permission_category, resolve_tool_policy

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def make_namespaced_tool_name(server_name: str, tool_name: str) -> str:
    """Create collision-free `<server>__<tool>` identifiers valid for LLM tool schemas."""
    srv = _SLUG_RE.sub("_", (server_name or "srv").strip().lower()).strip("_") or "srv"
    tl = _SLUG_RE.sub("_", (tool_name or "tool").strip()).strip("_") or "tool"
    return f"{srv}__{tl}"


class MCPToolRegistry:
    """Thread-safe registry mapping `(user_id, namespaced_tool_name)` -> DiscoveredTool."""

    def __init__(self, permission_repo: Optional[PermissionRepository] = None) -> None:
        self._permission_repo = permission_repo
        self._lock = threading.RLock()
        self._tools_by_user: Dict[str, Dict[str, DiscoveredTool]] = {}

    def register_server_tools(
        self,
        user_id: str,
        server_id: str,
        server_name: str,
        raw_tools: Sequence[object],
    ) -> List[DiscoveredTool]:
        """Replace tools for `(user_id, server_id)` with freshly discovered MCP tools."""
        with self._lock:
            user_map = self._tools_by_user.setdefault(user_id, {})
            # Drop stale tools from this server first
            for key in [k for k, v in user_map.items() if v.server_id == server_id]:
                user_map.pop(key, None)

            overrides_by_tool = (
                {
                    p.tool_name: p
                    for p in self._permission_repo.list_for_user(user_id)
                    if p.server_id == server_id
                }
                if self._permission_repo is not None
                else {}
            )

            discovered: List[DiscoveredTool] = []
            for raw in raw_tools:
                orig_name = str(getattr(raw, "name", ""))
                if not orig_name:
                    continue
                ns_name = make_namespaced_tool_name(server_name, orig_name)
                if ns_name in user_map:
                    ns_name = f"{ns_name}_{server_id[:6]}"

                annotations = getattr(raw, "annotations", None)
                read_only = getattr(annotations, "readOnlyHint", None) if annotations else None
                destructive = getattr(annotations, "destructiveHint", None) if annotations else None
                if read_only is None and annotations is not None:
                    read_only = getattr(annotations, "read_only_hint", None)
                if destructive is None and annotations is not None:
                    destructive = getattr(annotations, "destructive_hint", None)

                raw_schema = getattr(raw, "input_schema", None)
                if raw_schema is None:
                    raw_schema = getattr(raw, "inputSchema", {}) or {}
                schema = dict(raw_schema) if isinstance(raw_schema, dict) else {}

                inferred_cat = infer_permission_category(
                    orig_name,
                    read_only_hint=read_only,
                    destructive_hint=destructive,
                )
                override = overrides_by_tool.get(orig_name)
                base_tool = DiscoveredTool(
                    namespaced_name=ns_name,
                    original_name=orig_name,
                    server_id=server_id,
                    server_name=server_name,
                    user_id=user_id,
                    description=str(getattr(raw, "description", "") or "").strip(),
                    input_schema=schema,
                    read_only_hint=read_only,
                    destructive_hint=destructive,
                    permission_category=inferred_cat,
                    auto_approve=(inferred_cat == "READ_ONLY"),
                )
                effective_cat, requires_approval = resolve_tool_policy(base_tool, override)
                is_enabled = bool(override.enabled) if override is not None else True
                final_tool = DiscoveredTool(
                    namespaced_name=base_tool.namespaced_name,
                    original_name=base_tool.original_name,
                    server_id=base_tool.server_id,
                    server_name=base_tool.server_name,
                    user_id=base_tool.user_id,
                    description=base_tool.description,
                    input_schema=base_tool.input_schema,
                    read_only_hint=base_tool.read_only_hint,
                    destructive_hint=base_tool.destructive_hint,
                    permission_category=effective_cat,
                    auto_approve=not requires_approval,
                    enabled=is_enabled,
                )
                user_map[ns_name] = final_tool
                discovered.append(final_tool)

            return discovered

    def set_tool_enabled(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        enabled: bool,
    ) -> Optional[DiscoveredTool]:
        """Persist and update enablement state for a single tool without dropping the MCP session."""
        with self._lock:
            user_map = self._tools_by_user.get(user_id, {})
            updated_tool: Optional[DiscoveredTool] = None
            target_key = None
            for ns_name, existing in user_map.items():
                if existing.server_id == server_id and existing.original_name == tool_name:
                    target_key = ns_name
                    updated_tool = DiscoveredTool(
                        namespaced_name=existing.namespaced_name,
                        original_name=existing.original_name,
                        server_id=existing.server_id,
                        server_name=existing.server_name,
                        user_id=existing.user_id,
                        description=existing.description,
                        input_schema=existing.input_schema,
                        read_only_hint=existing.read_only_hint,
                        destructive_hint=existing.destructive_hint,
                        permission_category=existing.permission_category,
                        auto_approve=existing.auto_approve,
                        enabled=bool(enabled),
                    )
                    break

            if self._permission_repo is not None:
                default_cat = updated_tool.permission_category if updated_tool else "MODIFICATION"
                default_auto = updated_tool.auto_approve if updated_tool else False
                self._permission_repo.set_tool_enabled(
                    user_id=user_id,
                    server_id=server_id,
                    tool_name=tool_name,
                    enabled=bool(enabled),
                    default_category=default_cat,
                    default_auto_approve=default_auto,
                )

            if target_key and updated_tool is not None:
                user_map[target_key] = updated_tool
            return updated_tool

    def clear_server_tools(self, user_id: str, server_id: str) -> None:
        with self._lock:
            user_map = self._tools_by_user.get(user_id, {})
            for key in [k for k, v in user_map.items() if v.server_id == server_id]:
                user_map.pop(key, None)

    def list_tools(
        self,
        user_id: str,
        server_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[DiscoveredTool]:
        with self._lock:
            tools = list(self._tools_by_user.get(user_id, {}).values())
            if server_id:
                tools = [t for t in tools if t.server_id == server_id]
            if not include_disabled:
                tools = [t for t in tools if t.enabled]
            return sorted(tools, key=lambda t: t.namespaced_name)

    def get_tool(self, user_id: str, namespaced_name: str) -> Optional[DiscoveredTool]:
        with self._lock:
            return self._tools_by_user.get(user_id, {}).get(namespaced_name)
