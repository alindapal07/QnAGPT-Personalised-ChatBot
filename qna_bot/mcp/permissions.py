# Backward compatibility facade for qna_bot.mcp_client.permissions
import warnings
from qna_bot.mcp_client.permissions import *  # noqa: F401, F403

warnings.warn(
    "Importing from 'qna_bot.mcp' is deprecated; use 'qna_bot.mcp_client.permissions' instead.",
    DeprecationWarning,
    stacklevel=2,
)
