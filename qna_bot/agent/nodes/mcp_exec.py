"""MCP tool discovery, task planning, tool selection, permissions, HITL approval, and execution."""

from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Any, Dict, List
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import interrupt

from qna_bot.agent.mcp_adapter import build_mcp_langchain_tools, validate_tool_arguments
from qna_bot.agent.nodes.base import (
    BaseNodeSuite,
    _TaskPlanSchema,
    _collect_web_sources_from_messages,
    _executed_call_signatures_in_turn,
    _normalize_agent_tool_calls,
    _select_relevant_tools_for_binding,
    logger,
)
from qna_bot.agent.prompts import (
    AGENT_TOOL_SYSTEM_ADDENDUM,
    QNAGPT_SYSTEM_PROMPT,
    TASK_PLANNING_PROMPT,
)
from qna_bot.agent.state import AgentState
from qna_bot.database.models import AuditLogRecord
from qna_bot.mcp_client.security import format_masked_arguments, sanitize_tool_output


class MCPNodesMixin:
    """Mixin containing MCP tool discovery, planning, selection, permission, and execution nodes."""

    def discover_tools(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Discover currently authorized MCP and built-in tools for the active user."""
        user_id = state.get("user_id") or self.settings.default_user_id
        mcp_tools = self._list_mcp_tools_safe(user_id)
        catalog: List[Dict[str, Any]] = [
            {
                "name": t.namespaced_name,
                "original_name": t.original_name,
                "server_id": t.server_id,
                "server_name": t.server_name,
                "category": t.permission_category,
                "auto_approve": t.auto_approve,
                "description": t.description,
            }
            for t in mcp_tools
        ]
        for btool in self.tools:
            catalog.append(
                {
                    "name": btool.name,
                    "original_name": btool.name,
                    "server_id": "builtin",
                    "server_name": "QnAGPT Built-in",
                    "category": "READ_ONLY",
                    "auto_approve": True,
                    "description": btool.description,
                }
            )
        return {
            "available_tools": catalog,
            "agent_status": "Discovering tools",
            "activity": "Discovering available MCP tools...",
        }

    def plan_task(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Build a concise user-safe multi-step execution plan for the task."""
        user_query = (state.get("user_query") or "").strip()
        catalog = state.get("available_tools") or []
        if not self.api_key or not catalog:
            return {
                "planned_steps": ["Analyze request and respond with available context"],
                "agent_status": "Planning",
                "activity": "Planning the task...",
            }

        builtins = [item for item in catalog if item.get("server_id") == "builtin"]
        mcp_only = [item for item in catalog if item.get("server_id") != "builtin"]
        planning_catalog = builtins + mcp_only[: max(5, 30 - len(builtins))]
        tools_summary = "\n".join(
            f"- {item['name']} ({item['server_name']}, {item['category']}): {item['description'][:120]}"
            for item in planning_catalog
        )
        prompt = (
            f"{TASK_PLANNING_PROMPT}\n\n"
            f"Available Tools:\n{tools_summary}\n\n"
            f"User Request: {user_query}"
        )
        steps: List[str] = []
        try:
            parsed = self._make_llm(temperature=0.0, max_tokens=256).with_structured_output(
                _TaskPlanSchema
            ).invoke(prompt)
            steps = [s.strip() for s in (parsed.steps or []) if s.strip()][:4]
        except Exception:
            steps = ["Select appropriate tools", "Execute authorized actions", "Synthesize final answer"]

        return {
            "planned_steps": steps or ["Execute request"],
            "agent_status": "Planning",
            "activity": "Planning the task...",
        }

    def select_tools(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Use Agent LLM with tool_choice='auto' and dynamically bound MCP + built-in tools."""
        user_id = state.get("user_id") or self.settings.default_user_id
        user_query = (state.get("user_query") or "").strip()
        iteration = int(state.get("iteration_count", 0))
        max_iter = int(state.get("max_iterations", self.settings.max_agent_iterations))

        if not self.api_key:
            return {
                "answer": "GROQ_API_KEY is not configured. Please add it to your `.env` file.",
                "pending_tool_calls": [],
                "needs_more_steps": False,
                "agent_status": "Failed",
                "activity": "Preparing final response...",
            }

        if iteration >= max_iter:
            return {
                "pending_tool_calls": [],
                "needs_more_steps": False,
                "agent_status": "Processing results",
                "activity": "Preparing final response...",
            }

        mcp_lc_tools = (
            build_mcp_langchain_tools(self.mcp_manager, user_id=user_id)
            if self.mcp_manager is not None
            else []
        )
        combined_tools = _select_relevant_tools_for_binding(
            mcp_lc_tools,
            self.tools,
            user_query=user_query,
            planned_steps=state.get("planned_steps") or (),
        )

        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state.get("planned_steps") or []))
        sys_msg = SystemMessage(
            content=(
                QNAGPT_SYSTEM_PROMPT
                + AGENT_TOOL_SYSTEM_ADDENDUM
                + (f"\n\nTask Plan:\n{plan_text}" if plan_text else "")
            )
        )
        history = list(state.get("messages") or [])
        lc_messages: List[BaseMessage] = [sys_msg, *history[-16:]]
        if not any(
            isinstance(m, HumanMessage) and str(m.content).strip() == user_query
            for m in history[-3:]
        ):
            lc_messages.append(HumanMessage(content=user_query))

        try:
            response = self._invoke_agent_llm(
                lc_messages,
                combined_tools,
                allow_text_fallback=True,
            )
        except Exception as exc:
            logger.exception("Tool selection failed")
            return {
                "answer": f"Unable to plan tool execution due to model error: {exc}",
                "pending_tool_calls": [],
                "needs_more_steps": False,
                "agent_status": "Failed",
                "error": str(exc),
                "activity": "Preparing final response...",
            }

        raw_calls = list(getattr(response, "tool_calls", None) or [])
        valid_names = {t.name for t in combined_tools}
        normalized_calls = _normalize_agent_tool_calls(raw_calls, valid_names)
        executed_sigs = _executed_call_signatures_in_turn(history)
        fresh_calls = [
            tc
            for tc in normalized_calls
            if (tc["name"], json.dumps(tc["args"], sort_keys=True, default=str))
            not in executed_sigs
        ]
        if fresh_calls:
            normalized_msg = AIMessage(
                content=str(getattr(response, "content", "") or ""),
                tool_calls=[
                    {"id": tc["id"], "name": tc["name"], "args": tc["args"], "type": "tool_call"}
                    for tc in fresh_calls
                ],
            )
            return {
                "messages": [normalized_msg],
                "pending_tool_calls": fresh_calls,
                "pending_approval": None,
                "approval_decision": None,
                "needs_more_steps": True,
                "agent_status": "Executing",
                "activity": "Selecting tools...",
            }

        answer_text = str(getattr(response, "content", "") or "").strip()
        return {
            "answer": answer_text,
            "pending_tool_calls": [],
            "pending_approval": None,
            "needs_more_steps": False,
            "agent_status": "Processing results",
            "activity": "Preparing final response...",
        }

    def check_permissions(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Enforce backend tool permission policies and trigger approval requests when required."""
        user_id = state.get("user_id") or self.settings.default_user_id
        thread_id = state.get("thread_id") or "default_thread"
        pending_calls = list(state.get("pending_tool_calls") or [])
        if not pending_calls:
            return {"pending_approval": None, "activity": "Preparing final response..."}

        if self.mcp_manager is None:
            return {"pending_approval": None, "activity": "Executing tool..."}

        for call in pending_calls:
            tool_name = str(call.get("name", ""))
            args = dict(call.get("args") or {})
            if tool_name in self.builtin_tools_map:
                continue

            mcp_tool = self.mcp_manager.tool_registry.get_tool(user_id, tool_name)
            if mcp_tool is None:
                continue

            if mcp_tool.permission_category == "BLOCKED":
                if self.audit_repo is not None:
                    self.audit_repo.record(
                        AuditLogRecord(
                            execution_id=f"exec_{uuid.uuid4().hex[:10]}",
                            user_id=user_id,
                            thread_id=thread_id,
                            server_id=mcp_tool.server_id,
                            server_name=mcp_tool.server_name,
                            tool_name=mcp_tool.original_name,
                            status="blocked",
                            masked_arguments=format_masked_arguments(args),
                            result_summary="Execution blocked by security policy.",
                            started_at=time.time(),
                            completed_at=time.time(),
                            duration_ms=0.0,
                        )
                    )
                return {
                    "pending_approval": None,
                    "pending_tool_calls": pending_calls,
                    "error": f"Tool '{mcp_tool.original_name}' is blocked by security policy.",
                    "agent_status": "Failed",
                    "activity": "Checking tool permissions...",
                }

            if not mcp_tool.auto_approve and self.approval_manager is not None:
                reason = (
                    f"{mcp_tool.permission_category} operation '{mcp_tool.original_name}' "
                    f"on MCP server '{mcp_tool.server_name}' requires user approval."
                )
                record = self.approval_manager.create_pending_approval(
                    user_id=user_id,
                    thread_id=thread_id,
                    server_id=mcp_tool.server_id,
                    tool_name=mcp_tool.original_name,
                    arguments=args,
                    category=mcp_tool.permission_category,
                    reason=reason,
                )
                if self.audit_repo is not None:
                    self.audit_repo.record(
                        AuditLogRecord(
                            execution_id=f"exec_{uuid.uuid4().hex[:10]}",
                            user_id=user_id,
                            thread_id=thread_id,
                            server_id=mcp_tool.server_id,
                            server_name=mcp_tool.server_name,
                            tool_name=mcp_tool.original_name,
                            status="awaiting_approval",
                            masked_arguments=record.masked_arguments,
                            result_summary="Waiting for human approval.",
                            started_at=time.time(),
                        )
                    )
                return {
                    "pending_approval": {
                        "approval_id": record.approval_id,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "server_id": mcp_tool.server_id,
                        "server_name": mcp_tool.server_name,
                        "tool_name": mcp_tool.original_name,
                        "namespaced_name": mcp_tool.namespaced_name,
                        "arguments_hash": record.arguments_hash,
                        "masked_arguments": record.masked_arguments,
                        "category": record.category,
                        "reason": record.reason,
                    },
                    "agent_status": "Waiting for approval",
                    "activity": "Waiting for approval...",
                }

        return {
            "pending_approval": None,
            "agent_status": "Executing",
            "activity": "Executing tool...",
        }

    def request_approval(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Pause graph execution via LangGraph interrupt() until human approves or rejects."""
        payload = state.get("pending_approval") or {}
        decision = interrupt(payload)
        normalized = (
            dict(decision)
            if isinstance(decision, dict)
            else {"approved": bool(decision)}
        )
        return {
            "approval_decision": normalized,
            "agent_status": "Executing" if normalized.get("approved") else "Failed",
            "activity": "Processing approval decision...",
        }

    def _run_single_tool_call(
        self: BaseNodeSuite,
        call: Dict[str, Any],
        user_id: str,
        thread_id: str,
        approval_id: str,
    ) -> tuple[ToolMessage, Dict[str, Any]]:
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        tool_name = str(call.get("name") or "")
        args = dict(call.get("args") or {})
        started = time.time()
        exec_id = f"exec_{uuid.uuid4().hex[:10]}"

        if tool_name in self.builtin_tools_map:
            btool = self.builtin_tools_map[tool_name]
            try:
                out = sanitize_tool_output(str(btool.invoke(args)))
                dur = round((time.time() - started) * 1000.0, 2)
                event = {
                    "execution_id": exec_id,
                    "server_name": "QnAGPT Built-in",
                    "tool_name": tool_name,
                    "status": "completed",
                    "summary": out[:280],
                    "duration_ms": dur,
                    "timestamp": started,
                }
                return ToolMessage(content=out, tool_call_id=call_id, name=tool_name), event
            except Exception as exc:
                dur = round((time.time() - started) * 1000.0, 2)
                err_txt = f"Built-in tool '{tool_name}' failed: {exc}"
                event = {
                    "execution_id": exec_id,
                    "server_name": "QnAGPT Built-in",
                    "tool_name": tool_name,
                    "status": "failed",
                    "summary": err_txt,
                    "duration_ms": dur,
                    "timestamp": started,
                }
                return ToolMessage(content=err_txt, tool_call_id=call_id, name=tool_name), event

        if self.mcp_manager is None:
            msg = f"MCP client manager is unavailable to run '{tool_name}'."
            return ToolMessage(content=msg, tool_call_id=call_id, name=tool_name), {
                "execution_id": exec_id,
                "server_name": "MCP",
                "tool_name": tool_name,
                "status": "failed",
                "summary": msg,
                "duration_ms": 0.0,
                "timestamp": started,
            }

        mcp_tool = self.mcp_manager.tool_registry.get_tool(user_id, tool_name)
        if mcp_tool is None:
            msg = f"Tool '{tool_name}' is not registered or authorized for user '{user_id}'."
            return ToolMessage(content=msg, tool_call_id=call_id, name=tool_name), {
                "execution_id": exec_id,
                "server_name": "MCP",
                "tool_name": tool_name,
                "status": "failed",
                "summary": msg,
                "duration_ms": 0.0,
                "timestamp": started,
            }

        masked_args = format_masked_arguments(args)
        if mcp_tool.permission_category == "BLOCKED":
            msg = f"Tool '{mcp_tool.original_name}' is blocked by security policy."
            return ToolMessage(content=msg, tool_call_id=call_id, name=tool_name), {
                "execution_id": exec_id,
                "server_name": mcp_tool.server_name,
                "tool_name": mcp_tool.original_name,
                "status": "blocked",
                "summary": msg,
                "duration_ms": 0.0,
                "timestamp": started,
            }

        # Backend HITL approval verification (cannot be bypassed by frontend)
        if not mcp_tool.auto_approve:
            verified = (
                self.approval_manager is not None
                and bool(approval_id)
                and self.approval_manager.verify_and_consume(
                    approval_id=approval_id,
                    user_id=user_id,
                    thread_id=thread_id,
                    server_id=mcp_tool.server_id,
                    tool_name=mcp_tool.original_name,
                    arguments=args,
                )
            )
            if not verified:
                msg = (
                    f"Execution of '{mcp_tool.original_name}' refused: valid backend approval "
                    "matching exact arguments was not granted."
                )
                if self.audit_repo is not None:
                    self.audit_repo.record(
                        AuditLogRecord(
                            execution_id=exec_id,
                            user_id=user_id,
                            thread_id=thread_id,
                            server_id=mcp_tool.server_id,
                            server_name=mcp_tool.server_name,
                            tool_name=mcp_tool.original_name,
                            status="rejected",
                            masked_arguments=masked_args,
                            result_summary=msg,
                            started_at=started,
                            completed_at=time.time(),
                            duration_ms=0.0,
                        )
                    )
                return ToolMessage(content=msg, tool_call_id=call_id, name=tool_name), {
                    "execution_id": exec_id,
                    "server_name": mcp_tool.server_name,
                    "tool_name": mcp_tool.original_name,
                    "status": "rejected",
                    "summary": msg,
                    "duration_ms": 0.0,
                    "timestamp": started,
                }

        try:
            validated_args = validate_tool_arguments(mcp_tool, args)
            result = self.mcp_manager.execute_tool(
                user_id=user_id,
                namespaced_tool_name=tool_name,
                arguments=validated_args,
                timeout=self.settings.mcp_tool_timeout_sec,
            )
            status_str = "completed" if result.success else "failed"
            if self.audit_repo is not None:
                self.audit_repo.record(
                    AuditLogRecord(
                        execution_id=exec_id,
                        user_id=user_id,
                        thread_id=thread_id,
                        server_id=mcp_tool.server_id,
                        server_name=mcp_tool.server_name,
                        tool_name=mcp_tool.original_name,
                        status=status_str,
                        masked_arguments=masked_args,
                        result_summary=result.content[:300],
                        error=result.error,
                        started_at=started,
                        completed_at=time.time(),
                        duration_ms=result.duration_ms,
                    )
                )
            event = {
                "execution_id": exec_id,
                "server_name": mcp_tool.server_name,
                "tool_name": mcp_tool.original_name,
                "status": status_str,
                "summary": result.content[:280],
                "duration_ms": result.duration_ms,
                "timestamp": started,
            }
            return (
                ToolMessage(content=result.content or "Done.", tool_call_id=call_id, name=tool_name),
                event,
            )
        except Exception as exc:
            dur = round((time.time() - started) * 1000.0, 2)
            err_msg = f"Tool '{mcp_tool.original_name}' execution failed: {exc}"
            if self.audit_repo is not None:
                self.audit_repo.record(
                    AuditLogRecord(
                        execution_id=exec_id,
                        user_id=user_id,
                        thread_id=thread_id,
                        server_id=mcp_tool.server_id,
                        server_name=mcp_tool.server_name,
                        tool_name=mcp_tool.original_name,
                        status="failed",
                        masked_arguments=masked_args,
                        result_summary=err_msg,
                        error=str(exc),
                        started_at=started,
                        completed_at=time.time(),
                        duration_ms=dur,
                    )
                )
            event = {
                "execution_id": exec_id,
                "server_name": mcp_tool.server_name,
                "tool_name": mcp_tool.original_name,
                "status": "failed",
                "summary": err_msg,
                "duration_ms": dur,
                "timestamp": started,
            }
            return ToolMessage(content=err_msg, tool_call_id=call_id, name=tool_name), event

    def execute_tools(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Execute authorized MCP/built-in tool calls (parallelizing safe read-only calls)."""
        user_id = state.get("user_id") or self.settings.default_user_id
        thread_id = state.get("thread_id") or "default_thread"
        pending_calls = list(state.get("pending_tool_calls") or [])
        approval = state.get("pending_approval") or {}
        decision = state.get("approval_decision") or {}
        approval_id = str(decision.get("approval_id") or approval.get("approval_id") or "")

        all_read_only = True
        if self.mcp_manager is not None:
            for call in pending_calls:
                cname = str(call.get("name") or "")
                if cname in self.builtin_tools_map:
                    continue
                tool = self.mcp_manager.tool_registry.get_tool(user_id, cname)
                if tool is None or tool.permission_category != "READ_ONLY":
                    all_read_only = False
                    break

        tool_messages: List[ToolMessage] = []
        new_events: List[Dict[str, Any]] = []

        if all_read_only and len(pending_calls) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(pending_calls))) as pool:
                futures = [
                    pool.submit(self._run_single_tool_call, call, user_id, thread_id, "")
                    for call in pending_calls
                ]
                for fut in futures:
                    msg, ev = fut.result()
                    tool_messages.append(msg)
                    new_events.append(ev)
        else:
            for call in pending_calls:
                msg, ev = self._run_single_tool_call(call, user_id, thread_id, approval_id)
                tool_messages.append(msg)
                new_events.append(ev)

        combined_events = list(state.get("tool_executions") or []) + new_events
        updated_web_sources = _collect_web_sources_from_messages(
            tool_messages,
            existing_sources=state.get("web_sources") or [],
        )
        return {
            "messages": tool_messages,
            "tool_executions": combined_events,
            "web_sources": updated_web_sources,
            "pending_tool_calls": [],
            "pending_approval": None,
            "approval_decision": None,
            "activity": "Processing tool results...",
            "agent_status": "Processing results",
        }

    def observe_results(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Observe tool outcomes or handle rejected/blocked tool calls cleanly."""
        pending_calls = list(state.get("pending_tool_calls") or [])
        decision = state.get("approval_decision") or {}
        approval = state.get("pending_approval") or {}
        user_id = state.get("user_id") or self.settings.default_user_id
        thread_id = state.get("thread_id") or "default_thread"

        extra_messages: List[ToolMessage] = []
        new_events: List[Dict[str, Any]] = []
        rejected = bool(approval) and not bool(decision.get("approved"))

        if pending_calls:
            for call in pending_calls:
                cid = str(call.get("id") or "")
                cname = str(call.get("name") or "tool")
                reason = (
                    f"Action '{cname}' was rejected by the user and was NOT executed."
                    if rejected
                    else (state.get("error") or f"Action '{cname}' did not execute.")
                )
                extra_messages.append(ToolMessage(content=reason, tool_call_id=cid, name=cname))
                new_events.append(
                    {
                        "execution_id": f"exec_{uuid.uuid4().hex[:10]}",
                        "server_name": approval.get("server_name", "MCP"),
                        "tool_name": approval.get("tool_name", cname),
                        "status": "rejected" if rejected else "blocked",
                        "summary": reason,
                        "duration_ms": 0.0,
                        "timestamp": time.time(),
                    }
                )
            if rejected and self.audit_repo is not None and approval:
                self.audit_repo.record(
                    AuditLogRecord(
                        execution_id=f"exec_{uuid.uuid4().hex[:10]}",
                        user_id=user_id,
                        thread_id=thread_id,
                        server_id=str(approval.get("server_id", "")),
                        server_name=str(approval.get("server_name", "MCP")),
                        tool_name=str(approval.get("tool_name", "")),
                        status="rejected",
                        masked_arguments=str(approval.get("masked_arguments", "{}")),
                        result_summary="Rejected by user.",
                        started_at=time.time(),
                        completed_at=time.time(),
                        duration_ms=0.0,
                    )
                )

        iteration = int(state.get("iteration_count", 0)) + 1
        combined_events = list(state.get("tool_executions") or []) + new_events
        updates: Dict[str, Any] = {
            "iteration_count": iteration,
            "tool_executions": combined_events,
            "pending_tool_calls": [],
            "pending_approval": None,
            "approval_decision": None,
            "activity": "Processing tool results...",
            "agent_status": "Processing results",
        }
        if extra_messages:
            updates["messages"] = extra_messages
        if rejected:
            updates["needs_more_steps"] = False
        return updates

    def evaluate_progress(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Determine whether to loop for additional tool calls or finalize the response."""
        iteration = int(state.get("iteration_count", 0))
        max_iter = int(state.get("max_iterations", self.settings.max_agent_iterations))
        already_answered = bool((state.get("answer") or "").strip()) and not state.get("tool_executions")
        needs_more = bool(state.get("needs_more_steps", False)) and iteration < max_iter and not already_answered

        return {
            "needs_more_steps": needs_more,
            "activity": "Planning next step..." if needs_more else "Preparing final response...",
        }
