"""CLI launcher for starting the Streamlit QnA Chatbot."""

from pathlib import Path
import sys
from typing import List, Optional

from streamlit.web import cli as stcli


def main(argv: Optional[List[str]] = None) -> int:
    """Launch the Streamlit app via `qna-bot` or `python -m qna_bot`."""
    app_path = Path(__file__).resolve().parent / "app.py"
    extra_args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
