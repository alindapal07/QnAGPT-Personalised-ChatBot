"""Query analysis and query rewriting nodes for LangGraph workflow."""

from typing import Any, Dict, Optional

from qna_bot.agent.nodes.base import (
    BaseNodeSuite,
    _QueryAnalysisSchema,
    _extract_json_model,
    _recent_history_text,
    _requires_builtin_web_browse_loop,
    logger,
)
from qna_bot.agent.prompts import (
    QUERY_ANALYSIS_PROMPT,
    QUERY_REWRITE_PROMPT,
)
from qna_bot.agent.state import AgentState
from qna_bot.rag.crag import CRAGEvaluation


class IntentNodesMixin:
    """Mixin containing query analysis and query rewriting nodes."""

    def analyze_query(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Classify user query into GENERAL, DOCUMENT, WEB_SEARCH, FOLLOW_UP, or AGENT_TASK."""
        query = (state.get("user_query") or "").strip()
        chat_mode = state.get("chat_mode", "Auto (Agentic)")
        user_id = state.get("user_id") or self.settings.default_user_id
        max_iter = int(state.get("max_iterations") or self.settings.max_agent_iterations)

        if chat_mode == "General Chat":
            return self._turn_reset_fields(
                query_type="GENERAL",
                doc_required=False,
                web_required=False,
                activity="Preparing response...",
                max_iterations=max_iter,
            )

        user_docs = self._list_user_docs_safe(user_id)
        has_docs = bool(user_docs)
        doc_names = ", ".join(str(d.get("filename", "")) for d in user_docs[:10]) or "None"

        if chat_mode == "Document Chat":
            return self._turn_reset_fields(
                query_type="DOCUMENT",
                doc_required=True,
                web_required=False,
                activity="Analyzing document question...",
                max_iterations=max_iter,
            )

        mcp_tools = self._list_mcp_tools_safe(user_id)
        if chat_mode == "Agent Workspace":
            return self._turn_reset_fields(
                query_type="AGENT_TASK",
                doc_required=False,
                web_required=False,
                activity="Understanding your request...",
                max_iterations=max_iter,
            )

        mcp_names = ", ".join(t.namespaced_name for t in mcp_tools[:20]) or "None"
        history_txt = _recent_history_text(state.get("messages") or [])
        prompt = (
            QUERY_ANALYSIS_PROMPT.format(
                has_indexed_docs=str(has_docs).lower(),
                indexed_doc_names=doc_names,
                mcp_tool_count=len(mcp_tools),
                mcp_tool_names=mcp_names,
            )
            + f"\n\nRecent Conversation:\n{history_txt}\n\nLatest User Query: {query}"
        )

        parsed: Optional[_QueryAnalysisSchema] = None
        if self.api_key:
            try:
                llm = self._make_llm(temperature=0.0, max_tokens=512)
                parsed = llm.with_structured_output(_QueryAnalysisSchema).invoke(prompt)
            except Exception as exc:
                logger.warning("Structured query analysis failed, trying JSON fallback: %s", exc)
                try:
                    raw = self._make_llm(temperature=0.0, max_tokens=512).invoke(
                        prompt
                        + '\nRespond ONLY with JSON: {"query_type": "GENERAL|DOCUMENT|WEB_SEARCH|FOLLOW_UP|AGENT_TASK", "document_search_required": false, "web_search_required": false, "reason": "..."}'
                    )
                    parsed = _extract_json_model(str(getattr(raw, "content", "")), _QueryAnalysisSchema)
                except Exception as inner_exc:
                    logger.warning("Fallback query analysis failed: %s", inner_exc)

        if parsed is None:
            q_type = "AGENT_TASK" if mcp_tools else ("DOCUMENT" if has_docs else "GENERAL")
            return self._turn_reset_fields(
                query_type=q_type,
                doc_required=has_docs and not mcp_tools,
                web_required=False,
                activity="Understanding your request...",
                max_iterations=max_iter,
            )

        doc_req = bool(parsed.document_search_required)
        web_req = bool(parsed.web_search_required)
        if parsed.query_type == "DOCUMENT":
            doc_req = True
        elif parsed.query_type == "WEB_SEARCH":
            web_req = True
        elif parsed.query_type == "FOLLOW_UP" and has_docs and not web_req and not mcp_tools:
            doc_req = True
        elif parsed.query_type == "AGENT_TASK" and not mcp_tools:
            if not _requires_builtin_web_browse_loop(query):
                parsed.query_type = "DOCUMENT" if doc_req and has_docs else ("WEB_SEARCH" if web_req else "GENERAL")

        if doc_req and not has_docs:
            doc_req = False
            if parsed.query_type == "DOCUMENT":
                parsed.query_type = "GENERAL"

        return self._turn_reset_fields(
            query_type=parsed.query_type,
            doc_required=doc_req,
            web_required=web_req,
            activity="Understanding your request...",
            max_iterations=max_iter,
        )

    def rewrite_query(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Rewrite ambiguous or follow-up queries into a standalone semantic query."""
        original = (state.get("user_query") or "").strip()
        prev_rewrite = (state.get("rewritten_query") or "").strip()
        retry_count = int(state.get("retry_count", 0))

        if getattr(self.settings, "crag_enabled", True):
            evaluations = state.get("document_evaluations") or []
            last_eval: Optional[CRAGEvaluation] = None
            if evaluations:
                try:
                    last_eval = CRAGEvaluation.model_validate(evaluations[-1])
                except Exception:
                    pass
            prev_rewrites = [prev_rewrite] if prev_rewrite else []
            rewritten_obj = self.crag_pipeline.rewrite_query(
                query=original,
                messages=state.get("messages") or [],
                previous_rewrites=prev_rewrites,
                attempt=retry_count + 1,
                evaluation=last_eval,
            )
            return {
                "rewritten_query": rewritten_obj.rewritten_query,
                "activity": f"Refining search query: {rewritten_obj.rewritten_query[:60]}...",
            }

        history_txt = _recent_history_text(state.get("messages") or [])

        if not self.api_key:
            return {"rewritten_query": original, "activity": "Preparing document search..."}

        extra_instruction = ""
        if retry_count > 0 and prev_rewrite:
            extra_instruction = (
                f"\nPrevious search query '{prev_rewrite}' returned insufficient matches. "
                "Rephrase with broader core concepts and synonyms from the user's request."
            )

        prompt = (
            f"{QUERY_REWRITE_PROMPT}{extra_instruction}\n\n"
            f"Recent Conversation:\n{history_txt}\n\n"
            f"Current Query: {original}\n\n"
            "Standalone Search Query:"
        )
        try:
            response = self._make_llm(temperature=0.0, max_tokens=160).invoke(prompt)
            rewritten = str(getattr(response, "content", "") or "").strip().strip('"')
            if not rewritten:
                rewritten = original
        except Exception as exc:
            logger.warning("Query rewriting failed; using original query: %s", exc)
            rewritten = original

        return {"rewritten_query": rewritten, "activity": "Refining search query..."}
