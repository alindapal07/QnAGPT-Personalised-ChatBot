"""Safe adapter converting discovered MCP JSON Schemas into LangChain StructuredTools."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from qna_bot.core.logging_config import get_logger
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.models import DiscoveredTool

logger = get_logger(__name__)

_JSON_TYPE_MAP: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _resolve_field_type(prop_schema: Dict[str, Any]) -> Any:
    raw_type = prop_schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else "string"
    return _JSON_TYPE_MAP.get(str(raw_type or "string"), Any)


def build_pydantic_args_schema(
    tool_name: str,
    input_schema: Dict[str, Any],
) -> Type[BaseModel]:
    """Safely build a validated Pydantic v2 argument model from a JSON Schema dict."""
    if not isinstance(input_schema, dict):
        return create_model(f"{tool_name}Args")

    properties = input_schema.get("properties") or {}
    if not isinstance(properties, dict):
        return create_model(f"{tool_name}Args")

    required_set = set(input_schema.get("required") or [])
    fields: Dict[str, Tuple[Any, Any]] = {}

    for raw_field, raw_meta in properties.items():
        field_name = str(raw_field).strip()
        if not field_name or not field_name.isidentifier():
            continue
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        py_type = _resolve_field_type(meta)
        desc = str(meta.get("description") or "").strip()

        if field_name in required_set and "default" not in meta:
            fields[field_name] = (py_type, Field(..., description=desc))
        else:
            default_val = meta.get("default", None)
            annotated_type = Optional[py_type] if default_val is None else py_type
            fields[field_name] = (annotated_type, Field(default=default_val, description=desc))

    safe_model_name = "".join(ch if ch.isalnum() else "_" for ch in tool_name) + "_Schema"
    return create_model(safe_model_name, **fields)  # type: ignore[call-overload]


def validate_tool_arguments(
    tool: DiscoveredTool,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate incoming LLM tool arguments against the MCP tool's JSON schema."""
    schema_cls = build_pydantic_args_schema(tool.namespaced_name, tool.input_schema)
    validated = schema_cls.model_validate(arguments or {})
    return {k: v for k, v in validated.model_dump().items() if v is not None}


def build_mcp_langchain_tools(
    mcp_manager: MCPClientManager,
    user_id: str,
    tools: Optional[Sequence[DiscoveredTool]] = None,
) -> List[StructuredTool]:
    """Convert authorized user MCP tools into LangChain StructuredTools for LLM tool binding."""
    active_tools = list(tools) if tools is not None else mcp_manager.list_available_tools(user_id)
    lc_tools: List[StructuredTool] = []

    for mcp_tool in active_tools:
        if not mcp_tool.enabled or mcp_tool.permission_category == "BLOCKED":
            continue

        args_schema = build_pydantic_args_schema(
            mcp_tool.namespaced_name,
            mcp_tool.input_schema,
        )

        def _make_runner(target_tool: DiscoveredTool):
            def _run_tool(**kwargs: Any) -> str:
                res = mcp_manager.execute_tool(
                    user_id=user_id,
                    namespaced_tool_name=target_tool.namespaced_name,
                    arguments=kwargs,
                )
                return res.content

            return _run_tool

        desc = (
            f"[{mcp_tool.server_name} | {mcp_tool.permission_category}] "
            f"{mcp_tool.description or mcp_tool.original_name}"
        )
        lc_tool = StructuredTool.from_function(
            func=_make_runner(mcp_tool),
            name=mcp_tool.namespaced_name,
            description=desc,
            args_schema=args_schema,
            metadata={
                "mcp_server_id": mcp_tool.server_id,
                "mcp_server_name": mcp_tool.server_name,
                "mcp_original_tool": mcp_tool.original_name,
                "permission_category": mcp_tool.permission_category,
                "auto_approve": mcp_tool.auto_approve,
            },
        )
        lc_tools.append(lc_tool)

    return lc_tools
