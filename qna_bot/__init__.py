"""Enterprise QnA Chatbot package built with Streamlit, LangChain, and Groq."""

import importlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Tuple

__version__ = "0.1.0"

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AIAgent": ("qna_bot.services.agent_service", "AIAgent"),
    "AppSettings": ("qna_bot.config.settings", "AppSettings"),
    "get_settings": ("qna_bot.config.settings", "get_settings"),
    "load_settings": ("qna_bot.config.settings", "load_settings"),
}

__all__ = [
    "AIAgent",
    "AppSettings",
    "__version__",
    "get_settings",
    "load_settings",
    "resolve_lazy_export",
    "sanitize_import_path",
]


def sanitize_import_path() -> None:
    """Keep project root importable and prevent `qna_bot/` from shadowing PyPI packages (e.g. `mcp`)."""
    cleaned: List[str] = []
    project_present = False
    for entry in sys.path:
        try:
            resolved = (Path.cwd() if entry == "" else Path(entry)).resolve()
        except (OSError, ValueError):
            cleaned.append(entry)
            continue
        if resolved == _PACKAGE_DIR:
            continue
        if resolved == _PROJECT_ROOT:
            project_present = True
        cleaned.append(entry)

    if not project_present:
        cleaned.insert(0, str(_PROJECT_ROOT))
    sys.path[:] = cleaned

    # Clear shadowed top-level `mcp` if another runner imported `qna_bot/mcp` as `mcp`.
    mcp_mod = sys.modules.get("mcp")
    mcp_file = getattr(mcp_mod, "__file__", None) if mcp_mod is not None else None
    if mcp_file:
        try:
            if Path(mcp_file).resolve().is_relative_to(_PACKAGE_DIR / "mcp"):
                for mod_name in [k for k in list(sys.modules) if k == "mcp" or k.startswith("mcp.")]:
                    sys.modules.pop(mod_name, None)
        except (OSError, ValueError):
            pass


def resolve_lazy_export(
    package_name: str,
    namespace: Dict[str, Any],
    exports: Mapping[str, Tuple[str, str]],
    name: str,
) -> Any:
    """Resolve and cache a lazy package export without triggering circular barrel imports."""
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    namespace[name] = value
    return value


sanitize_import_path()


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, globals(), _EXPORTS, name)


def __dir__() -> List[str]:
    return sorted(__all__)
