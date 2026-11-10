"""Theme-aware CSS and dynamic light/dark detector loader."""

from functools import lru_cache
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def load_theme_markup() -> str:
    """Read and cache the CSS stylesheet and theme-detection script."""
    css = (_ASSETS_DIR / "styles.css").read_text(encoding="utf-8")
    js = (_ASSETS_DIR / "theme.js").read_text(encoding="utf-8")
    return f"<style>\n{css}\n</style>\n<script>\n{js}\n</script>"


def apply_styles() -> None:
    """Inject theme-aware CSS and dynamic theme detection script into Streamlit."""
    st.markdown(load_theme_markup(), unsafe_allow_html=True)
