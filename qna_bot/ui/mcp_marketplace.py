"""MCP Marketplace directory browser, connector details, OAuth PKCE / secret setup, and tool controls."""

import html
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.database.models import MCPServerRecord
from qna_bot.mcp.exceptions import MCPError
from qna_bot.mcp.models import MCPServerStatus
from qna_bot.mcp.registry.models import RegistryListing, TransportOption
from qna_bot.services.mcp_service import MCPService
from qna_bot.ui.design_system import (
    render_empty_state,
    render_html,
    render_metric_strip,
    render_page_header,
    status_pill_html,
)


def _safe_http_url(url: str) -> str:
    cleaned = (url or "").strip()
    if cleaned.startswith(("https://", "http://")):
        return cleaned
    return ""


def _status_pill(installed: Optional[Tuple[MCPServerRecord, MCPServerStatus]]) -> str:
    if installed is None:
        return status_pill_html("Available", "neutral")
    _, status = installed
    if status.connected:
        return status_pill_html("Connected", "ok")
    if status.status == "error":
        return status_pill_html("Error", "err")
    return status_pill_html("Disconnected", "warn")


def _render_icon_html(title: str, icon_url: str) -> str:
    letter = html.escape((title or "M").strip()[:1].upper() or "M")
    safe_icon = _safe_http_url(icon_url)
    if safe_icon and urlparse(safe_icon).scheme == "https":
        esc_url = html.escape(safe_icon, quote=True)
        return (
            f'<img src="{esc_url}" alt="{letter}" class="mkt-icon-img" '
            f'onerror="this.style.display=\'none\'"/>'
        )
    return f'<div class="mkt-icon-badge">{letter}</div>'


def _installed_record_to_listing(
    rec: MCPServerRecord,
    status: MCPServerStatus,
) -> RegistryListing:
    """Create a normalized RegistryListing view for an already-installed MCP server."""
    opt = TransportOption(
        transport=rec.transport,
        label=f"{rec.transport}",
        url=rec.url,
        command=rec.command,
        args=tuple(rec.args),
        package_type=rec.package_type or rec.transport,
        install_hint=rec.install_hint or rec.url or f"{rec.command} {' '.join(rec.args)}".strip(),
        direct_connectable=True,
    )
    preview_tools = tuple(
        {
            "name": str(t.get("original_name") or ""),
            "description": str(t.get("description") or ""),
            "inputSchema": dict(t.get("input_schema") or {}),
        }
        for t in (rec.discovered_tools or [])
        if t.get("original_name")
    )
    attr_name = (
        "Glama MCP Directory"
        if rec.registry_source == "glama"
        else ("Official MCP Registry" if rec.registry_source == "official_mcp" else "Installed Connector")
    )
    attr_url = (
        "https://glama.ai"
        if rec.registry_source == "glama"
        else ("https://registry.modelcontextprotocol.io" if rec.registry_source == "official_mcp" else "")
    )
    return RegistryListing(
        connector_id=f"{rec.registry_source}::{rec.registry_id or rec.server_id}",
        registry_source=rec.registry_source,
        registry_id=rec.registry_id or rec.server_id,
        name=rec.registry_id or rec.name,
        title=rec.name,
        description=rec.description or f"Installed {rec.transport} MCP server.",
        category=rec.category or "General",
        categories=(rec.category or "General",),
        icon_url=rec.icon_url,
        listing_url=rec.listing_url,
        attribution_name=attr_name,
        attribution_url=attr_url,
        transport_options=(opt,),
        auth_methods=(rec.auth_type or "none",),
        preview_tools=preview_tools,
        direct_connectable=True,
    )


def _render_attribution_banner(attributions: List[tuple[str, str]]) -> None:
    parts: List[str] = []
    for name, url in attributions:
        safe_url = _safe_http_url(url)
        if safe_url:
            parts.append(
                f'<a href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener">'
                f"{html.escape(name)}</a>"
            )
        else:
            parts.append(html.escape(name))
    if not parts:
        return
    render_html(
        f'<div class="mkt-attribution">Connector metadata powered by {" &nbsp;|&nbsp; ".join(parts)}. '
        f"Always review transport permissions &amp; credentials before connecting third-party servers.</div>"
    )


def _render_connect_panel(
    user_id: str,
    listing: RegistryListing,
    installed: Optional[Tuple[MCPServerRecord, MCPServerStatus]],
    mcp_service: MCPService,
) -> None:
    st.markdown("### Connection & Authentication")
    rec = installed[0] if installed else None
    status = installed[1] if installed else None

    if status and status.connected:
        c_s1, c_s2, c_s3 = st.columns([2, 1, 1])
        with c_s1:
            st.success(f"Connected ({status.tool_count} tools, {status.resource_count} resources)")
        with c_s2:
            if st.button("Refresh Tools", key=f"det_ref_{status.server_id}", use_container_width=True):
                try:
                    caps = mcp_service.refresh(status.server_id, user_id)
                    st.toast(f"Refreshed {caps['tools']} tools.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with c_s3:
            if st.button("Disconnect", key=f"det_disc_{status.server_id}", use_container_width=True):
                mcp_service.disconnect(status.server_id, user_id)
                st.rerun()
        return

    if not listing.direct_connectable or not listing.transport_options:
        st.warning(
            listing.connectability_reason
            or "This directory listing does not publish a directly connectable endpoint or runnable package."
        )
        if _safe_http_url(listing.repository_url):
            st.markdown(f"Review installation guide: [{listing.repository_url}]({listing.repository_url})")
        return

    connectable_opts = [opt for opt in listing.transport_options if opt.direct_connectable]
    if not connectable_opts:
        connectable_opts = list(listing.transport_options)

    opt_labels = [f"{opt.label} — {opt.install_hint or opt.url}" for opt in connectable_opts]
    opt_idx = st.selectbox(
        "Connection Transport",
        options=list(range(len(connectable_opts))),
        format_func=lambda i: opt_labels[i],
        key=f"det_opt_{listing.connector_id}",
    )
    selected_opt = connectable_opts[opt_idx]

    if not selected_opt.direct_connectable:
        st.error(selected_opt.notes or "Selected transport option cannot be connected directly.")
        return

    # ── Remote Transport Flow (Streamable HTTP / SSE) ─────────────────────────
    if selected_opt.transport in ("streamable_http", "sse"):
        st.caption(f"Endpoint: `{selected_opt.url}` | Transport: `{selected_opt.transport}`")
        default_auth = rec.auth_type if rec else (listing.auth_methods[0] if listing.auth_methods else "none")
        auth_choices = ["none", "oauth", "bearer_token", "api_key", "bearer_env", "api_key_env"]
        auth_idx = auth_choices.index(default_auth) if default_auth in auth_choices else 0
        auth_type = st.selectbox(
            "Authentication Method",
            options=auth_choices,
            index=auth_idx,
            format_func=lambda x: {
                "none": "No Authentication",
                "oauth": "OAuth 2.1 (Authorization Code + PKCE)",
                "bearer_token": "Bearer Token (Encrypted at Rest)",
                "api_key": "API Key (Encrypted at Rest)",
                "bearer_env": "Bearer Token (Env Var Reference)",
                "api_key_env": "API Key (Env Var Reference)",
            }[x],
            key=f"det_auth_{listing.connector_id}",
        )

        allow_insecure = st.checkbox(
            "Allow HTTP (non-TLS endpoint)",
            value=rec.allow_insecure_http if rec else selected_opt.url.startswith("http://"),
            key=f"det_http_{listing.connector_id}",
        )
        allow_private = st.checkbox(
            "Allow localhost / private network target (controlled local policy)",
            value=rec.allow_private_network if rec else False,
            key=f"det_priv_{listing.connector_id}",
        )

        header_name = selected_opt.required_headers[0] if selected_opt.required_headers else (
            "X-API-Key" if auth_type in ("api_key", "api_key_env") else "Authorization"
        )
        secret_val = ""
        env_var_ref = ""

        if auth_type in ("bearer_token", "api_key"):
            header_name = st.text_input(
                "HTTP Header Name",
                value=rec.auth_header_name if rec else header_name,
                key=f"det_hdr_{listing.connector_id}",
            )
            secret_val = st.text_input(
                "Secret Token / API Key (encrypted immediately via Fernet at rest)",
                type="password",
                key=f"det_sec_{listing.connector_id}",
            )
        elif auth_type in ("bearer_env", "api_key_env"):
            header_name = st.text_input(
                "HTTP Header Name",
                value=rec.auth_header_name if rec else header_name,
                key=f"det_ehdr_{listing.connector_id}",
            )
            env_var_ref = st.text_input(
                "Environment Variable Name",
                value=rec.auth_env_var if rec else "",
                placeholder="MCP_SERVER_TOKEN",
                key=f"det_evar_{listing.connector_id}",
            )

        if auth_type == "oauth":
            target_sid = rec.server_id if rec else None
            if rec is None:
                if st.button("Register OAuth Server Entry", key=f"det_reg_oauth_{listing.connector_id}"):
                    try:
                        created, _ = mcp_service.install_from_listing(
                            user_id=user_id,
                            listing=listing,
                            transport_option=selected_opt,
                            auth_type="oauth",
                            allow_insecure_http=allow_insecure,
                            allow_private_network=allow_private,
                            auto_connect=False,
                        )
                        st.success(f"Registered `{created.name}`. Start OAuth authorization below.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                return

            cred_status = mcp_service.get_credential_status(user_id, rec.server_id)
            if cred_status.get("configured") and not cred_status.get("expired"):
                st.info("Encrypted OAuth token is stored cleanly for this workspace.")
                ca1, ca2, ca3 = st.columns(3)
                with ca1:
                    if st.button("Connect Now", type="primary", key=f"det_oauth_conn_{rec.server_id}", use_container_width=True):
                        try:
                            mcp_service.connect(rec.server_id, user_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with ca2:
                    if st.button("Refresh Token", key=f"det_oauth_ref_{rec.server_id}", use_container_width=True):
                        try:
                            mcp_service.refresh_oauth_token(user_id, rec.server_id)
                            st.toast("OAuth token refreshed.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with ca3:
                    if st.button("Revoke Token", key=f"det_oauth_rev_{rec.server_id}", use_container_width=True):
                        mcp_service.revoke_credentials(user_id, rec.server_id)
                        st.toast("OAuth credentials revoked.")
                        st.rerun()
            else:
                with st.expander("OAuth Client & Endpoint Options (optional if `.well-known` supported)"):
                    oauth_cid = st.text_input("Client ID", key=f"oauth_cid_{rec.server_id}")
                    oauth_csec = st.text_input("Client Secret (optional)", type="password", key=f"oauth_cs_{rec.server_id}")
                    oauth_auth_ep = st.text_input("Authorization Endpoint URL (optional)", key=f"oauth_aep_{rec.server_id}")
                    oauth_tok_ep = st.text_input("Token Endpoint URL (optional)", key=f"oauth_tep_{rec.server_id}")
                    oauth_scopes = st.text_input("Scopes (space-separated)", key=f"oauth_scp_{rec.server_id}")

                if st.button("Start OAuth PKCE Sign-In", type="primary", key=f"det_oauth_start_{rec.server_id}"):
                    try:
                        sess = mcp_service.start_oauth_flow(
                            user_id=user_id,
                            server_id=rec.server_id,
                            client_id=oauth_cid,
                            client_secret=oauth_csec,
                            authorization_endpoint=oauth_auth_ep,
                            token_endpoint=oauth_tok_ep,
                            scopes=oauth_scopes,
                        )
                        st.session_state[f"oauth_pending_{rec.server_id}"] = sess.safe_public_view()
                    except Exception as exc:
                        st.error(str(exc))

                pending_view = st.session_state.get(f"oauth_pending_{rec.server_id}")
                if pending_view:
                    st.markdown(
                        f"**Step 1:** Open authorization URL: "
                        f"[{pending_view['authorization_endpoint']}]({pending_view['authorization_url']})"
                    )
                    cb_input = st.text_input(
                        "Step 2: Paste full callback redirect URL (with `?code=...&state=...`) or code",
                        value=str(st.query_params.get("code") or ""),
                        key=f"oauth_cb_{rec.server_id}",
                    )
                    cb_state = st.text_input(
                        "State (only needed if pasting raw code instead of redirect URL)",
                        value=str(st.query_params.get("state") or pending_view.get("state", "")),
                        key=f"oauth_st_{rec.server_id}",
                    )
                    if st.button("Verify State, Save Token & Connect", type="primary", key=f"det_oauth_fin_{rec.server_id}"):
                        try:
                            mcp_service.complete_oauth_flow(
                                user_id=user_id,
                                server_id=rec.server_id,
                                callback_url_or_code=cb_input,
                                state=cb_state or None,
                            )
                            st.session_state.pop(f"oauth_pending_{rec.server_id}", None)
                            mcp_service.connect(rec.server_id, user_id)
                            st.success("OAuth authorized & MCP server connected!")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
            return

        btn_label = "Save Credentials & Connect" if auth_type in ("bearer_token", "api_key") else "One-Click Connect"
        if st.button(btn_label, type="primary", key=f"det_conn_remote_{listing.connector_id}"):
            try:
                _, st_out = mcp_service.install_from_listing(
                    user_id=user_id,
                    listing=listing,
                    transport_option=selected_opt,
                    auth_type=auth_type,
                    bearer_token=secret_val if auth_type == "bearer_token" else "",
                    api_key=secret_val if auth_type == "api_key" else "",
                    auth_header_name=header_name,
                    auth_env_var=env_var_ref,
                    allow_insecure_http=allow_insecure,
                    allow_private_network=allow_private,
                    auto_connect=True,
                )
                st.success(f"Connected ({st_out.tool_count if st_out else 0} tools discovered).")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return

    # ── Controlled Local STDIO Launch Flow ────────────────────────────────────
    cmd_preview = f"{selected_opt.command} {' '.join(selected_opt.args)}".strip()
    st.code(cmd_preview, language="bash")
    runtime_ok, runtime_msg = mcp_service.check_stdio_runtime(selected_opt.command)
    st.caption(runtime_msg)

    env_vars = list(selected_opt.env_variables or listing.env_variables)
    env_secrets: Dict[str, str] = {}
    if env_vars:
        st.markdown("**Required / Optional Environment Variables (stored encrypted at rest)**")
        for ev in env_vars:
            ev_name = str(ev.get("name") or "").strip()
            if not ev_name:
                continue
            req_tag = " *(required)*" if ev.get("required") else ""
            val = st.text_input(
                f"{ev_name}{req_tag}",
                type="password" if ev.get("isSecret", True) else "default",
                help=str(ev.get("description") or ""),
                key=f"det_env_{listing.connector_id}_{ev_name}",
            )
            if val.strip():
                env_secrets[ev_name] = val.strip()

    roots_str = st.text_input(
        "Allowed Filesystem Roots (optional, comma-separated)",
        key=f"det_roots_{listing.connector_id}",
    )
    allowed_roots = [r.strip() for r in roots_str.split(",") if r.strip()]

    approved_stdio = st.checkbox(
        "I have reviewed the local STDIO command and approve launching it on this machine.",
        value=False,
        key=f"det_stdio_ok_{listing.connector_id}",
    )

    col_launch, col_install = st.columns(2)
    with col_launch:
        if st.button("Approve & Connect Local Server", type="primary", key=f"det_conn_stdio_{listing.connector_id}", use_container_width=True):
            try:
                _, st_out = mcp_service.install_from_listing(
                    user_id=user_id,
                    listing=listing,
                    transport_option=selected_opt,
                    auth_type="none",
                    env_secrets=env_secrets,
                    allowed_roots=allowed_roots,
                    auto_connect=True,
                    stdio_confirmed=approved_stdio,
                )
                st.success(f"Connected local server ({st_out.tool_count if st_out else 0} tools).")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with col_install:
        if selected_opt.package_identifier and selected_opt.package_type in ("npm", "pypi", "oci", "docker"):
            if st.button("Pre-Install Package (Approved)", key=f"det_preinst_{listing.connector_id}", disabled=not approved_stdio, use_container_width=True):
                try:
                    ok, log_out = mcp_service.install_stdio_package(
                        package_type=selected_opt.package_type,
                        identifier=selected_opt.package_identifier,
                        version=selected_opt.version,
                        confirmed=approved_stdio,
                    )
                    if ok:
                        st.success("Package installed cleanly.")
                    else:
                        st.error("Package installation failed.")
                    if log_out:
                        st.code(log_out)
                except Exception as exc:
                    st.error(str(exc))


def _render_connector_details(
    user_id: str,
    listing: RegistryListing,
    installed: Optional[Tuple[MCPServerRecord, MCPServerStatus]],
    mcp_service: MCPService,
) -> None:
    if st.button("← Back to MCP Marketplace", key="btn_mkt_back"):
        st.session_state["mkt_selected_connector"] = None
        st.rerun()

    pill_html = _status_pill(installed)
    icon_html = _render_icon_html(listing.title, listing.icon_url)
    render_html(
        f'<div class="mkt-card-header">'
        f"{icon_html}"
        f'<div><div class="mkt-title">{html.escape(listing.title)} &nbsp; {pill_html}</div>'
        f'<div class="mkt-subtitle"><code>{html.escape(listing.name)}</code> &middot; {html.escape(listing.category)}</div></div>'
        f"</div>"
    )
    if listing.description:
        st.write(listing.description)

    meta_links: List[str] = []
    if _safe_http_url(listing.listing_url):
        safe_l = html.escape(listing.listing_url, quote=True)
        src_label = html.escape(listing.attribution_name or listing.registry_source)
        meta_links.append(f'<a class="mkt-link" href="{safe_l}" target="_blank" rel="noopener">View on {src_label}</a>')
    if _safe_http_url(listing.repository_url):
        safe_r = html.escape(listing.repository_url, quote=True)
        meta_links.append(f'<a class="mkt-link" href="{safe_r}" target="_blank" rel="noopener">Repository</a>')
    if listing.license_name:
        meta_links.append(f"License: <code>{html.escape(listing.license_name)}</code>")
    if listing.version:
        meta_links.append(f"Version: <code>{html.escape(listing.version)}</code>")
    if meta_links:
        render_html(f'<div style="font-size:12.5px;color:#64748B;">{" &nbsp;|&nbsp; ".join(meta_links)}</div>')

    st.divider()
    _render_connect_panel(user_id, listing, installed, mcp_service)
    st.divider()

    rec = installed[0] if installed else None
    tools = mcp_service.list_tools_with_persisted(user_id, rec.server_id) if rec else []
    resources = mcp_service.list_resources(user_id, rec.server_id) if rec else []
    prompts = mcp_service.list_prompts(user_id, rec.server_id) if rec else []

    tab_tools, tab_res, tab_prm = st.tabs(
        [
            f"Tools ({len(tools) if rec else len(listing.preview_tools)})",
            f"Resources ({len(resources)})",
            f"Prompts ({len(prompts)})",
        ]
    )

    with tab_tools:
        if rec and tools:
            det_tool_q = st.text_input(
                "Filter connector tools",
                placeholder="Search tool name or description...",
                key=f"det_tool_q_{rec.server_id}",
            ).strip().lower()
            filtered_det_tools = [
                t
                for t in tools
                if not det_tool_q
                or det_tool_q in t.original_name.lower()
                or det_tool_q in t.description.lower()
            ]
            if len(filtered_det_tools) > 50 and not det_tool_q:
                st.caption(f"Showing first 50 of {len(filtered_det_tools)} tools — search above to filter.")
                filtered_det_tools = filtered_det_tools[:50]
            categories = ("READ_ONLY", "MODIFICATION", "SENSITIVE", "BLOCKED")
            for tool in filtered_det_tools:
                with st.container(border=True):
                    c_head, c_tgl = st.columns([4, 1])
                    with c_head:
                        st.markdown(f"**{tool.original_name}** (`{tool.permission_category}`)")
                        if tool.description:
                            st.caption(tool.description)
                    with c_tgl:
                        new_en = st.checkbox(
                            "Enabled",
                            value=tool.enabled,
                            key=f"det_tool_en_{rec.server_id}_{tool.original_name}",
                        )
                        if new_en != tool.enabled:
                            mcp_service.set_tool_enabled(user_id, rec.server_id, tool.original_name, new_en)
                            st.rerun()
                    if tool.input_schema:
                        with st.expander("Input Schema", expanded=False):
                            st.json(tool.input_schema)
                    p1, p2, p3 = st.columns([2, 1.5, 1])
                    with p1:
                        new_cat = st.selectbox(
                            "Permission",
                            options=list(categories),
                            index=categories.index(tool.permission_category),
                            key=f"det_tcat_{rec.server_id}_{tool.original_name}",
                        )
                    with p2:
                        new_auto = st.checkbox(
                            "Auto-approve",
                            value=tool.auto_approve,
                            disabled=(new_cat in ("SENSITIVE", "BLOCKED")),
                            key=f"det_tauto_{rec.server_id}_{tool.original_name}",
                        )
                    with p3:
                        if st.button("Save", key=f"det_tsave_{rec.server_id}_{tool.original_name}", use_container_width=True):
                            mcp_service.set_tool_permission(
                                user_id=user_id,
                                server_id=rec.server_id,
                                tool_name=tool.original_name,
                                category=new_cat,  # type: ignore[arg-type]
                                auto_approve=(new_auto and new_cat not in ("SENSITIVE", "BLOCKED")),
                                enabled=new_en,
                            )
                            st.toast("Tool policy saved.")
                            st.rerun()
        elif listing.preview_tools:
            st.caption("Registry capability preview (connect server to verify live tool schemas):")
            for pt in listing.preview_tools:
                with st.expander(str(pt.get("name") or "tool"), expanded=False):
                    st.write(pt.get("description") or "No description provided.")
                    if pt.get("inputSchema"):
                        st.json(pt["inputSchema"])
        else:
            st.info("Connect this MCP server to dynamically discover its live tools, schemas, and controls.")

    with tab_res:
        if not resources:
            st.info("No resources discovered or server is disconnected.")
        for res in resources:
            with st.expander(f"{res.name} ({res.mime_type})", expanded=False):
                st.caption(f"URI: `{res.uri}`")
                if st.button("Read Resource", key=f"det_rres_{res.server_id}_{res.uri}"):
                    try:
                        st.code(mcp_service.read_resource(user_id, res.server_id, res.uri)["content"])
                    except Exception as exc:
                        st.error(str(exc))

    with tab_prm:
        if not prompts:
            st.info("No prompts discovered or server is disconnected.")
        for prm in prompts:
            with st.expander(prm.name, expanded=False):
                if prm.description:
                    st.write(prm.description)


def render_mcp_marketplace_page(
    settings: AppSettings,
    user_id: str,
    mcp_service: MCPService,
) -> None:
    """Render the light-mode MCP Connectors Marketplace and Details workflow."""
    installed_map = mcp_service.map_installed_connectors(user_id=user_id)

    selected_listing: Optional[RegistryListing] = st.session_state.get("mkt_selected_connector")
    if selected_listing is not None:
        installed = (
            installed_map.get(selected_listing.connector_id)
            or installed_map.get(selected_listing.registry_id)
            or installed_map.get(selected_listing.title.lower())
        )
        _render_connector_details(user_id, selected_listing, installed, mcp_service)
        return

    h_col, ref_col = st.columns([4.5, 1.2])
    with h_col:
        render_page_header(
            "MCP Marketplace",
            "Discover connectors from the Official MCP Registry and Glama Directory, authenticate securely, and bind tools to QnAGPT.",
        )
    with ref_col:
        if st.button("Refresh", icon=":material/refresh:", key="btn_mkt_refresh", use_container_width=True):
            mcp_service.clear_registry_cache()
            st.session_state["mkt_cursor_stack"] = []
            st.rerun()

    c_src, c_search, c_cat, c_tr = st.columns([1.4, 2.2, 1.3, 1.1])
    with c_src:
        source_choice = st.selectbox(
            "Directory Source",
            options=["all", "official_mcp", "glama"],
            format_func=lambda x: {
                "all": "All Directories",
                "official_mcp": "Official MCP Registry",
                "glama": "Glama Directory",
            }[x],
            key="mkt_source_sel",
        )
    with c_search:
        query = st.text_input(
            "Search Connectors",
            placeholder="Search name, description, category, or tool...",
            key="mkt_search_q",
        )
    with c_cat:
        cat_filter = st.selectbox(
            "Category",
            options=[
                "All",
                "Developer Tools",
                "Data & Databases",
                "Web & Search",
                "Files & Storage",
                "Productivity",
                "Communication",
                "Cloud & DevOps",
                "AI & Utilities",
                "General",
            ],
            key="mkt_cat_sel",
        )
    with c_tr:
        tr_filter = st.selectbox(
            "Transport",
            options=["All", "streamable_http", "sse", "stdio"],
            key="mkt_tr_sel",
        )

    feed_view = st.selectbox(
        "View Filter",
        options=["all", "recommended", "popular", "recent", "installed", "connected", "disconnected"],
        format_func=lambda x: {
            "all": "All Connectors",
            "recommended": "Recommended",
            "popular": "Popular",
            "recent": "Recently Added",
            "installed": "Installed",
            "connected": "Connected",
            "disconnected": "Disconnected",
        }[x],
        key="mkt_feed_sel",
    )

    cursor_stack: List[str] = list(st.session_state.get("mkt_cursor_stack") or [])
    current_cursor = cursor_stack[-1] if cursor_stack else None

    statuses = mcp_service.list_statuses(user_id=user_id)
    records = mcp_service.list_servers(user_id=user_id)
    conn_count = sum(1 for s in statuses if s.connected)

    if feed_view in ("installed", "connected", "disconnected"):
        listings: List[RegistryListing] = []
        status_by_id = {s.server_id: s for s in statuses}
        for rec in records:
            st_obj = status_by_id.get(rec.server_id)
            if st_obj is None:
                continue
            if feed_view == "connected" and not st_obj.connected:
                continue
            if feed_view == "disconnected" and st_obj.connected:
                continue
            item = _installed_record_to_listing(rec, st_obj)
            if mcp_service.registry_service._matches_filters(
                item,
                query=query,
                category=cat_filter,
                transport_filter=tr_filter,
                only_connectable=False,
            ):
                listings.append(item)
        attributions = [
            ("Official MCP Registry", settings.official_registry_url),
            ("Glama MCP Directory", "https://glama.ai"),
        ]
        page_warning = ""
        has_next = False
        next_cursor = None
    else:
        page = mcp_service.discover_marketplace(
            source=source_choice,
            query=query,
            category=cat_filter,
            feed=feed_view,
            transport_filter=tr_filter,
            limit=20,
            cursor=current_cursor,
        )
        listings = page.items
        attributions = page.attributions
        page_warning = page.warning
        has_next = page.has_next_page
        next_cursor = page.next_cursor

    render_metric_strip(
        [
            ("Showing", len(listings)),
            ("Installed", len(records)),
            ("Connected", conn_count),
            ("Disconnected", max(0, len(records) - conn_count)),
        ]
    )

    _render_attribution_banner(attributions)
    if page_warning:
        st.info(page_warning)

    if not listings:
        render_empty_state(
            "No matching MCP connectors",
            "Try clearing the search query or selecting another category, transport, or directory view.",
        )
        return

    cols = st.columns(2)
    for idx, item in enumerate(listings):
        with cols[idx % 2]:
            inst = (
                installed_map.get(item.connector_id)
                or installed_map.get(item.registry_id)
                or installed_map.get(item.title.lower())
            )
            pill_html = _status_pill(inst)
            icon_html = _render_icon_html(item.title, item.icon_url)
            tool_cnt = inst[1].tool_count if inst else (item.tool_count or len(item.preview_tools))
            auth_label = ", ".join(item.auth_methods)
            short_desc = html.escape((item.description or "No description provided.")[:135])
            listing_link_html = ""
            if _safe_http_url(item.listing_url):
                src_tag = "Glama" if item.registry_source == "glama" else "Registry"
                listing_link_html = (
                    f'<a class="mkt-link" href="{html.escape(item.listing_url, quote=True)}" '
                    f'target="_blank" rel="noopener">{src_tag}</a>'
                )

            with st.container(border=True):
                render_html(
                    f'<div class="mkt-card-header">'
                    f"{icon_html}"
                    f'<div><div class="mkt-title">{html.escape(item.title)} &nbsp; {pill_html}</div>'
                    f'<div class="mkt-subtitle">{html.escape(item.category)}{(" &middot; " + listing_link_html) if listing_link_html else ""}</div></div>'
                    f"</div>"
                    f'<div class="mkt-card-desc">{short_desc}</div>'
                    f'<div class="mkt-tag-row">'
                    f'<span class="mkt-tag">{html.escape(item.primary_transport)}</span>'
                    f'<span class="mkt-tag">Auth: {html.escape(auth_label)}</span>'
                    f'<span class="mkt-tag">{tool_cnt} tools</span>'
                    f'<span class="mkt-tag">{html.escape(item.primary_install_info[:28])}</span>'
                    f"</div>"
                )
                b_det, b_act = st.columns(2)
                with b_det:
                    if st.button("Details", key=f"mkt_det_{idx}_{item.connector_id}", use_container_width=True):
                        st.session_state["mkt_selected_connector"] = item
                        st.rerun()
                with b_act:
                    if inst is not None and inst[1].connected:
                        if st.button("Disconnect", key=f"mkt_disc_{idx}_{item.connector_id}", use_container_width=True):
                            mcp_service.disconnect(inst[0].server_id, user_id)
                            st.rerun()
                    elif inst is not None:
                        if st.button("Connect", type="primary", key=f"mkt_reconn_{idx}_{item.connector_id}", use_container_width=True):
                            try:
                                mcp_service.connect(inst[0].server_id, user_id)
                                st.rerun()
                            except Exception:
                                st.session_state["mkt_selected_connector"] = item
                                st.rerun()
                    else:
                        act_label = "Connect" if item.direct_connectable else "Setup Guide"
                        if st.button(
                            act_label,
                            type="primary" if item.direct_connectable else "secondary",
                            key=f"mkt_conn_{idx}_{item.connector_id}",
                            use_container_width=True,
                        ):
                            st.session_state["mkt_selected_connector"] = item
                            st.rerun()

    if cursor_stack or (has_next and next_cursor):
        st.divider()
        p_prev, p_next = st.columns([1, 1])
        with p_prev:
            if st.button("Previous Page", disabled=(not cursor_stack), key="mkt_prev_page"):
                cursor_stack.pop()
                st.session_state["mkt_cursor_stack"] = cursor_stack
                st.rerun()
        with p_next:
            if st.button("Next Page", disabled=(not (has_next and next_cursor)), key="mkt_next_page"):
                if next_cursor:
                    cursor_stack.append(next_cursor)
                    st.session_state["mkt_cursor_stack"] = cursor_stack
                    st.rerun()
