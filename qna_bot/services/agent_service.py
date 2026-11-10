"""LangChain and ChatGroq AI orchestration service."""

import time
from typing import Any, Dict, Generator, Iterable, List, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from qna_bot.config.settings import DEFAULT_SYSTEM_PROMPT, get_settings
from qna_bot.core.exceptions import AgentExecutionError, ConfigurationError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)


def build_langchain_messages(
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
) -> List[BaseMessage]:
    """Convert session chat history into LangChain message objects."""
    lc_messages: List[BaseMessage] = []
    cleaned_system = system_prompt.strip()
    if cleaned_system:
        lc_messages.append(SystemMessage(content=cleaned_system))

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        img_b64 = msg.get("image_b64")
        img_mime = msg.get("image_mime") or "image/png"

        if role == "user":
            if img_b64:
                text_part = content.strip() or "Please analyze this image."
                parts: List[Dict[str, Any]] = [
                    {"type": "text", "text": text_part},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_mime};base64,{img_b64}"},
                    },
                ]
                lc_messages.append(HumanMessage(content=parts))
            else:
                lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return lc_messages


def _extract_chunk_text(chunk: Any) -> str:
    """Extract printable text from a LangChain chunk or raw string."""
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(text_parts)
    return ""


def stream_chunks(
    chunk_iterable: Iterable[Any],
    delay_sec: float = 0.02,
) -> Generator[str, None, None]:
    """Yield non-empty text tokens from a stream with an optional pacing delay."""
    for chunk in chunk_iterable:
        text = _extract_chunk_text(chunk)
        if text:
            if delay_sec > 0:
                time.sleep(delay_sec)
            yield text


class AIAgent:
    """Encapsulates ChatGroq model invocation and text/vision routing."""

    def __init__(
        self,
        text_model: str,
        vision_model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str = "",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        stream_delay_sec: float | None = None,
    ) -> None:
        settings = get_settings()
        self.text_model = text_model
        self.vision_model = vision_model or settings.default_vision_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = (api_key or settings.groq_api_key).strip()
        self.system_prompt = system_prompt
        self.stream_delay_sec = (
            settings.stream_chunk_delay_sec
            if stream_delay_sec is None
            else stream_delay_sec
        )

    def _make_llm(self, model_name: str) -> ChatGroq:
        return ChatGroq(
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )

    def active_model_name(self, has_image: bool = False) -> str:
        """Return the model identifier that will handle the request."""
        return self.vision_model if has_image else self.text_model

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        has_image: bool = False,
    ) -> Generator[str, None, None]:
        """Stream assistant tokens, automatically routing multimodal turns to vision."""
        if not self.api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        model_name = self.active_model_name(has_image=has_image)
        lc_messages = build_langchain_messages(messages, self.system_prompt)
        logger.info(
            "Streaming completion with model=%s (vision=%s, message_count=%d)",
            model_name,
            has_image,
            len(lc_messages),
        )

        try:
            llm = self._make_llm(model_name)
            yield from stream_chunks(
                llm.stream(lc_messages),
                delay_sec=self.stream_delay_sec,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.exception("Model streaming failed for model=%s", model_name)
            raise AgentExecutionError(str(exc)) from exc
