"""Enterprise MCP Server Management Dashboard, Capability Explorer, and Configuration UI."""

import html
import json
import time

import streamlit as st

from qna_bot.mcp.exceptions import MCPError
from qna_bot.services.mcp_service import MCPService
from qna_bot.ui.design_system import (
    render_empty_state,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)


def _parse_kv_lines(raw_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (raw_text or "").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        if key.strip():
            result[key.strip()] = val.strip()
    return result


def _status_pill(status: str, connected: bool) -> str:
    if connected:
        return status_pill_html("Connected", "ok")
    if status == "error":
        return status_pill_html("Error", "err")
    return status_pill_html("Disconnected", "warn")


def render_mcp_dashboard(user_id: str, mcp_service: MCPService) -> None:
    """Render Installed MCP Servers Management Dashboard & Capability Explorers."""
    render_page_header(
        "MCP Servers",
        "Connect, configure, test health, refresh tools, and manage credentials across local STDIO and remote HTTP MCP servers.",
    )

    records = {r.server_id: r for r in mcp_service.list_servers(user_id=user_id)}
    statuses = mcp_service.list_statuses(user_id=user_id)
    tools = mcp_service.list_tools(user_id=user_id, include_disabled=True)
    resources = mcp_service.list_resources(user_id=user_id)
    prompts = mcp_service.list_prompts(user_id=user_id)

    total_srv = len(statuses)
    conn_cnt = sum(1 for s in statuses if s.connected)
    err_cnt = sum(1 for s in statuses if s.status == "error")
    disc_cnt = max(0, total_srv - conn_cnt - err_cnt)

    render_metric_strip(
        [
            ("Total Servers", total_srv),
            ("Connected", conn_cnt),
            ("Disconnected", disc_cnt),
            ("Errors", err_cnt),
            ("Discovered Tools", len(tools)),
        ]
    )

    # ── Add Server Form ───────────────────────────────────────────────────────
    with st.expander("Add Custom MCP Server", icon=":material/add:", expanded=(total_srv == 0)):
        srv_name = st.text_input("Server Name", placeholder="e.g. filesystem or github")
        transport = st.selectbox(
            "Transport Type",
            options=["stdio", "streamable_http", "sse"],
            format_func=lambda x: {
                "stdio": "Local STDIO",
                "streamable_http": "Remote Streamable HTTP",
                "sse": "Legacy SSE (Compatibility)",
            }[x],
        )

        command = args_raw = env_raw = roots_raw = ""
        url = auth_header = auth_env_var = secret_input = ""
        auth_type = "none"
        allow_insecure = allow_private = False

        if transport == "stdio":
            command = st.text_input(
                "Command (allowlisted: python, uv, uvx, node, npx, docker)",
                placeholder="npx",
            )
            args_raw = st.text_input(
                "Arguments (comma or newline separated)",
                placeholder="-y, @modelcontextprotocol/server-filesystem, C:/path/to/folder",
            )
            roots_raw = st.text_input(
                "Allowed Filesystem Roots (optional, comma-separated)",
                placeholder="C:/Users/example/Documents",
            )
            env_raw = st.text_area(
                "Environment Variable References (KEY=env:VAR_NAME, one per line)",
                placeholder="GITHUB_TOKEN=env:MY_GITHUB_TOKEN",
                height=70,
            )
        else:
            url = st.text_input("Remote MCP Server URL", placeholder="https://example.com/mcp")
            auth_type = st.selectbox(
                "Authentication Method",
                options=["none", "bearer_token", "api_key", "oauth", "bearer_env", "api_key_env"],
            )
            if auth_type in ("bearer_token", "api_key"):
                auth_header = st.text_input(
                    "Header Name",
                    value="X-API-Key" if auth_type == "api_key" else "Authorization",
                )
                secret_input = st.text_input(
                    "Credential Value (encrypted with Fernet at rest)",
                    type="password",
                    placeholder="Paste token or API key...",
                )
            elif auth_type != "none":
                auth_header = st.text_input("Header Name", value="Authorization")
                auth_env_var = st.text_input(
                    "Token Environment Variable Name",
                    placeholder="MCP_SERVER_TOKEN",
                )
            c_sec1, c_sec2 = st.columns(2)
            with c_sec1:
                allow_insecure = st.checkbox("Allow HTTP (non-TLS)", value=False)
            with c_sec2:
                allow_private = st.checkbox("Allow local/private network (controlled policy)", value=False)

        enabled = st.checkbox("Enabled", value=True)
        if st.button("Register MCP Server", type="primary", key="btn_register_mcp"):
            parsed_args = [
                part.strip()
                for line in args_raw.splitlines()
                for part in line.split(",")
                if part.strip()
            ]
            parsed_roots = [r.strip() for r in roots_raw.split(",") if r.strip()]
            try:
                rec = mcp_service.register_server(
                    user_id=user_id,
                    name=srv_name,
                    transport=transport,
                    enabled=enabled,
                    command=command,
                    args=parsed_args,
                    env_refs=_parse_kv_lines(env_raw),
                    url=url,
                    auth_type=auth_type,
                    auth_header_name=auth_header,
                    auth_env_var=auth_env_var,
                    allow_insecure_http=allow_insecure,
                    allow_private_network=allow_private,
                    allowed_roots=parsed_roots,
                )
                if auth_type == "bearer_token" and secret_input.strip():
                    mcp_service.store_bearer_token(user_id, rec.server_id, secret_input, header_name=auth_header)
                elif auth_type == "api_key" and secret_input.strip():
                    mcp_service.store_api_key(user_id, rec.server_id, secret_input, header_name=auth_header)
                st.success(f"Registered MCP server '{rec.name}'.")
                st.rerun()
            except MCPError as exc:
                st.error(f"Configuration validation failed: {exc}")

    render_section_header("Configured Servers", f"{len(statuses)} registered")
    if not statuses:
        render_empty_state(
            "No MCP servers configured yet",
            "Browse the MCP Marketplace for one-click connectors or register a custom STDIO / Streamable HTTP server above.",
        )
        return

    c_flt, c_tr = st.columns([2, 1.5])
    with c_flt:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All", "Connected", "Disconnected", "Failed"],
            key="mcp_dash_filter",
        )
    with c_tr:
        tr_filter = st.selectbox(
            "Transport Filter",
            options=["All", "streamable_http", "stdio", "sse"],
            key="mcp_dash_tr_filter",
        )

    filtered_statuses = [
        s
        for s in statuses
        if (
            status_filter == "All"
            or (status_filter == "Connected" and s.connected)
            or (status_filter == "Disconnected" and not s.connected and s.status != "error")
            or (status_filter == "Failed" and s.status == "error")
        )
        and (tr_filter == "All" or s.transport == tr_filter)
    ]

    if not filtered_statuses:
        render_empty_state("No matching servers", "Adjust the status or transport filter above.")
    else:
        for status in filtered_statuses:
            sid = status.server_id
            rec = records.get(sid)
            pill = _status_pill(status.status, status.connected)
            last_conn = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(status.last_connected_at))
                if status.last_connected_at
                else "Never"
            )
            auth_badge = (
                "No Auth"
                if status.auth_type == "none"
                else ("Encrypted Credentials Ready" if status.credential_configured and not status.credential_expired else "Auth Required / Expired")
            )
            health_label = status.health_status.title() if status.health_status else "Unknown"
            endpoint_preview = (
                rec.url
                if rec and rec.url
                else (f"{rec.command} {' '.join(rec.args)}".strip() if rec else "")
            )
            listing_link = ""
            if status.listing_url and status.listing_url.startswith(("https://", "http://")):
                listing_link = (
                    f' &middot; <a class="mkt-link" href="{html.escape(status.listing_url, quote=True)}" '
                    f'target="_blank" rel="noopener">{html.escape(status.registry_source)} listing</a>'
                )

            with st.container(border=True):
                render_html(
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">'
                    f'<strong style="font-size:14px;color:#172033;">{html.escape(status.name)}</strong> '
                    f"{pill} {status_pill_html(status.transport, 'neutral')} "
                    f'<span style="font-size:12px;color:#64748B;"><code>{html.escape(status.registry_source)}</code>{listing_link}</span>'
                    f"</div>"
                )
                if endpoint_preview:
                    st.caption(f"Endpoint / Command: `{endpoint_preview}`")
                st.caption(
                    f"Tools: {status.tool_count}  |  Resources: {status.resource_count}  |  Prompts: {status.prompt_count}  |  "
                    f"Health: {health_label}  |  Auth: {auth_badge}  |  Last connected: {last_conn}"
                )
                if status.last_error:
                    st.error(status.last_error)

                b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
                with b1:
                    if status.connected:
                        if st.button("Disconnect", key=f"mcp_disc_{sid}", use_container_width=True):
                            mcp_service.disconnect(sid, user_id)
                            st.rerun()
                    else:
                        if st.button("Connect", key=f"mcp_conn_{sid}", type="primary", use_container_width=True):
                            try:
                                mcp_service.connect(sid, user_id)
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))
                with b2:
                    if st.button("Reconnect", key=f"mcp_reconn_{sid}", use_container_width=True):
                        try:
                            mcp_service.reconnect(sid, user_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with b3:
                    if st.button("Refresh", key=f"mcp_ref_{sid}", disabled=not status.connected, use_container_width=True):
                        try:
                            caps = mcp_service.refresh(sid, user_id)
                            st.toast(f"Refreshed: {caps['tools']} tools, {caps['resources']} resources.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with b4:
                    if st.button("Health", key=f"mcp_hlth_{sid}", disabled=not status.connected, use_container_width=True):
                        hlth = mcp_service.check_health(sid, user_id)
                        st.toast(f"Health: {hlth.health_status}")
                        st.rerun()
                with b5:
                    if st.button("Configure", key=f"mcp_cfg_{sid}", use_container_width=True):
                        st.session_state["mcp_configure_server_id"] = (
                            None if st.session_state.get("mcp_configure_server_id") == sid else sid
                        )
                        st.rerun()
                with b6:
                    if st.button("Explore", key=f"mcp_exp_{sid}", use_container_width=True):
                        st.session_state["mcp_explore_server_id"] = sid
                with b7:
                    if st.session_state.get("mcp_confirm_remove") != sid:
                        if st.button("Remove", key=f"mcp_rem_{sid}", use_container_width=True):
                            st.session_state["mcp_confirm_remove"] = sid
                            st.rerun()
                    else:
                        if st.button("Confirm", key=f"mcp_rem_yes_{sid}", type="primary", use_container_width=True):
                            mcp_service.remove_server(sid, user_id)
                            st.session_state["mcp_confirm_remove"] = None
                            st.rerun()

                if st.session_state.get("mcp_configure_server_id") == sid and rec is not None:
                    with st.expander(f"Configure {rec.name}", expanded=True):
                        if rec.transport in ("streamable_http", "sse"):
                            new_url = st.text_input("Endpoint URL", value=rec.url, key=f"cfg_url_{sid}")
                            new_tok = st.text_input("Update Encrypted Token / API Key (leave blank to keep)", type="password", key=f"cfg_tok_{sid}")
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                new_http = st.checkbox("Allow HTTP", value=rec.allow_insecure_http, key=f"cfg_http_{sid}")
                            with cc2:
                                new_priv = st.checkbox("Allow Private Network", value=rec.allow_private_network, key=f"cfg_priv_{sid}")
                            cs1, cs2 = st.columns(2)
                            with cs1:
                                if st.button("Save Changes", type="primary", key=f"cfg_save_{sid}"):
                                    mcp_service.register_server(
                                        user_id=user_id,
                                        name=rec.name,
                                        transport=rec.transport,
                                        enabled=rec.enabled,
                                        url=new_url,
                                        auth_type=rec.auth_type,
                                        auth_header_name=rec.auth_header_name,
                                        auth_env_var=rec.auth_env_var,
                                        allow_insecure_http=new_http,
                                        allow_private_network=new_priv,
                                        server_id=sid,
                                    )
                                    if new_tok.strip():
                                        if rec.auth_type == "api_key":
                                            mcp_service.store_api_key(user_id, sid, new_tok, header_name=rec.auth_header_name)
                                        else:
                                            mcp_service.store_bearer_token(user_id, sid, new_tok, header_name=rec.auth_header_name)
                                    st.session_state["mcp_configure_server_id"] = None
                                    st.success("Server configuration updated.")
                                    st.rerun()
                            with cs2:
                                if st.button("Revoke Stored Secrets", key=f"cfg_rev_{sid}"):
                                    mcp_service.revoke_credentials(user_id, sid)
                                    st.toast("Encrypted credentials deleted.")
                                    st.rerun()
                        else:
                            new_roots = st.text_input(
                                "Allowed Filesystem Roots (comma-separated)",
                                value=", ".join(rec.allowed_roots),
                                key=f"cfg_roots_{sid}",
                            )
                            if st.button("Save Local Policy", type="primary", key=f"cfg_lsave_{sid}"):
                                mcp_service.register_server(
                                    user_id=user_id,
                                    name=rec.name,
                                    transport=rec.transport,
                                    enabled=rec.enabled,
                                    command=rec.command,
                                    args=rec.args,
                                    env_refs=rec.env_refs,
                                    allowed_roots=[r.strip() for r in new_roots.split(",") if r.strip()],
                                    server_id=sid,
                                )
                                st.session_state["mcp_configure_server_id"] = None
                                st.success("Policy saved.")
                                st.rerun()

    # ── Capability Explorers ──────────────────────────────────────────────────
    render_section_header("Server Capabilities Explorer")
    explore_id = st.session_state.get("mcp_explore_server_id")
    offline_or_live_tools = (
        mcp_service.list_tools_with_persisted(user_id, explore_id)
        if explore_id
        else tools
    )
    tab_tools, tab_resources, tab_prompts = st.tabs(
        [f"Tools ({len(offline_or_live_tools)})", f"Resources ({len(resources)})", f"Prompts ({len(prompts)})"]
    )

    with tab_tools:
        if explore_id and st.button("Show All Servers", key="mcp_show_all_tools"):
            st.session_state["mcp_explore_server_id"] = None
            st.rerun()
        tool_filter = st.text_input(
            "Filter discovered tools",
            placeholder="Search tool name or description...",
            key="mcp_dash_tool_filter",
        ).strip().lower()
        visible_tools = [
            t
            for t in offline_or_live_tools
            if not tool_filter
            or tool_filter in t.original_name.lower()
            or tool_filter in t.description.lower()
            or tool_filter in t.server_name.lower()
        ]
        if not visible_tools:
            render_empty_state("No matching tools", "Connect a server above or clear the tool search filter.")
        else:
            if len(visible_tools) > 50 and not tool_filter:
                st.caption(f"Showing first 50 of {len(visible_tools)} discovered tools — search above to filter.")
                visible_tools = visible_tools[:50]
            for tool in visible_tools:
                state_tag = "Enabled" if tool.enabled else "Disabled"
                with st.expander(
                    f"{tool.server_name} :: {tool.original_name} ({tool.permission_category} · {state_tag})",
                    expanded=False,
                ):
                    t_col1, t_col2 = st.columns([4, 1])
                    with t_col1:
                        st.write(tool.description or "No description provided.")
                        st.caption(f"LangGraph Tool ID: `{tool.namespaced_name}` | Auto-approve: `{tool.auto_approve}`")
                    with t_col2:
                        new_en = st.checkbox(
                            "Enabled",
                            value=tool.enabled,
                            key=f"dash_tool_en_{tool.server_id}_{tool.original_name}",
                        )
                        if new_en != tool.enabled:
                            mcp_service.set_tool_enabled(user_id, tool.server_id, tool.original_name, new_en)
                            st.rerun()
                    st.json(tool.input_schema or {})

    with tab_resources:
        if not resources:
            render_empty_state("No MCP resources available", "Connect a server that exposes MCP resources to read them here.")
        for res in resources:
            with st.expander(f"{res.server_name} :: {res.name} ({res.mime_type})", expanded=False):
                st.caption(f"URI: `{res.uri}`")
                if res.description:
                    st.write(res.description)
                if st.button("Read Resource", key=f"read_res_{res.server_id}_{res.uri}"):
                    try:
                        payload = mcp_service.read_resource(user_id, res.server_id, res.uri)
                        st.code(payload["content"])
                    except Exception as exc:
                        st.error(str(exc))

    with tab_prompts:
        if not prompts:
            render_empty_state("No MCP prompts available", "Connect a server that publishes prompt templates to inspect them here.")
        for prm in prompts:
            with st.expander(f"{prm.server_name} :: {prm.name}", expanded=False):
                if prm.description:
                    st.write(prm.description)
                st.json(prm.arguments)
                if st.button("Inspect Prompt", key=f"get_prm_{prm.server_id}_{prm.name}"):
                    try:
                        out = mcp_service.get_prompt(user_id, prm.server_id, prm.name)
                        st.code(json.dumps(out, indent=2), language="json")
                    except Exception as exc:
                        st.error(str(exc))
