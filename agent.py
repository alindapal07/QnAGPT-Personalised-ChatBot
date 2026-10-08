"""
agent.py — All AI / LangChain / Groq logic lives here.
No CSS, no Streamlit UI, no HTML.
"""
import time
from typing import Any, Dict, Generator, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import DEFAULT_VISION_MODEL, GROQ_API_KEY


# ── Message conversion ────────────────────────────────────────────────────────

def build_langchain_messages(
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Any]:
    """
    Convert session message history into LangChain message objects.
    Handles text-only and multimodal (image + text) messages.
    """
    lc: List[Any] = []
    if system_prompt.strip():
        lc.append(SystemMessage(content=system_prompt.strip()))

    for msg in messages:
        role     = msg.get("role")
        content  = msg.get("content", "")
        img_b64  = msg.get("image_b64")
        img_mime = msg.get("image_mime", "image/png")

        if role == "user":
            if img_b64:
                text_part = content.strip() or "Please analyze this image."
                parts = [
                    {"type": "text", "text": text_part},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}},
                ]
                lc.append(HumanMessage(content=parts))
            else:
                lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))

    return lc


# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_chunks(chunk_iterable) -> Generator[str, None, None]:
    """Yield text content from LangChain stream chunks with a small delay."""
    for chunk in chunk_iterable:
        if hasattr(chunk, "content") and chunk.content:
            time.sleep(0.02)
            yield chunk.content
        elif isinstance(chunk, str) and chunk:
            time.sleep(0.02)
            yield chunk


# ── AIAgent ───────────────────────────────────────────────────────────────────

class AIAgent:
    """
    Encapsulates all AI model interactions.
    Text model for normal queries; vision model for image analysis.
    """

    def __init__(
        self,
        text_model: str,
        vision_model: str = DEFAULT_VISION_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str = "",
        system_prompt: str = "",
    ) -> None:
        self.text_model    = text_model
        self.vision_model  = vision_model
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.api_key       = api_key or GROQ_API_KEY
        self.system_prompt = system_prompt

    # ── Internal LLM factory ──────────────────────────────────────────────────

    def _make_llm(self, model_name: str) -> ChatGroq:
        return ChatGroq(
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )

    # ── Public streaming methods ──────────────────────────────────────────────

    def stream(
        self,
        messages: List[Dict[str, Any]],
        has_image: bool = False,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the appropriate model.
        Routes to vision_model when has_image=True.
        Raises RuntimeError on configuration or API errors.
        """
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        model_name = self.vision_model if has_image else self.text_model
        lc_messages = build_langchain_messages(messages, self.system_prompt)
        llm = self._make_llm(model_name)
        yield from stream_chunks(llm.stream(lc_messages))

    def active_model_name(self, has_image: bool = False) -> str:
        """Return the name of whichever model will be used."""
        return self.vision_model if has_image else self.text_model

