"""Light-only semantic stylesheet loader for QnAGPT Workspace."""

from functools import lru_cache
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def load_theme_markup() -> str:
    """Read and cache the single Light-theme CSS stylesheet."""
    css = (_ASSETS_DIR / "styles.css").read_text(encoding="utf-8")
    return f"<style>\n{css}\n</style>"


def apply_styles() -> None:
    """Inject the Light-theme design system into Streamlit."""
    st.markdown(load_theme_markup(), unsafe_allow_html=True)
