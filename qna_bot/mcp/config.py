# Backward compatibility facade for qna_bot.mcp_client.config
import warnings
from qna_bot.mcp_client.config import *  # noqa: F401, F403

warnings.warn(
    "Importing from 'qna_bot.mcp' is deprecated; use 'qna_bot.mcp_client.config' instead.",
    DeprecationWarning,
    stacklevel=2,
)
