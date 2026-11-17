# Backward compatibility facade for qna_bot.mcp_client.server_registry
import warnings
from qna_bot.mcp_client.server_registry import *  # noqa: F401, F403

warnings.warn(
    "Importing from 'qna_bot.mcp' is deprecated; use 'qna_bot.mcp_client.server_registry' instead.",
    DeprecationWarning,
    stacklevel=2,
)
