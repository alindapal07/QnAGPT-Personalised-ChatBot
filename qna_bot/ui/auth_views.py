"""Authentication UI (Sign In, Create Account, Google/GitHub OAuth) and Account Management page."""

import html
from pathlib import Path
from typing import Optional

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.core.exceptions import AuthenticationError, ConfigurationError, SecurityError
from qna_bot.rag.qdrant_store import QdrantVectorStoreManager, migrate_faiss_user_to_qdrant
from qna_bot.schemas.auth import AuthenticatedSession, AuthResult
from qna_bot.services.auth_service import AuthService
from qna_bot.services.user_service import UserService
from qna_bot.ui.design_system import (
    BRAND_SVG,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)


def _apply_auth_result(result: AuthResult, user_service: UserService) -> None:
    st.session_state["auth_session_token"] = result.session_token
    st.session_state["user_id"] = result.session.user.id
    prefs = user_service.get_settings(result.session.user.id)
    if prefs.get("text_model"):
        st.session_state["cfg_text_model"] = prefs["text_model"]
    if prefs.get("temperature"):
        try:
            st.session_state["cfg_temperature"] = float(prefs["temperature"])
        except ValueError:
            pass
    if prefs.get("max_tokens"):
        try:
            st.session_state["cfg_max_tokens"] = int(prefs["max_tokens"])
        except ValueError:
            pass
    if prefs.get("system_prompt"):
        st.session_state["cfg_system_prompt"] = prefs["system_prompt"]
    if prefs.get("chat_mode"):
        st.session_state["chat_mode"] = prefs["chat_mode"]


def handle_oauth_callback_params(
    auth_service: AuthService,
    user_service: UserService,
) -> Optional[str]:
    """Process OAuth redirect query parameters if present, returning error message on failure."""
    try:
        qp = st.query_params
        provider = str(qp.get("oauth_provider", "") or "").strip().lower()
        code = str(qp.get("code", "") or "").strip()
        state = str(qp.get("state", "") or "").strip()
        err = str(qp.get("error", "") or "").strip()
    except Exception:
        return None

    if not provider:
        return None

    try:
        st.query_params.clear()
    except Exception:
        pass

    if err:
        return f"{provider.title()} sign-in was cancelled or denied: {err}"
    if not code or not state:
        return f"Incomplete {provider.title()} callback parameters."

    try:
        res = auth_service.complete_oauth_login(provider=provider, code=code, state=state)
        _apply_auth_result(res, user_service)
        return None
    except Exception as exc:
        return str(exc)


def render_auth_gate(
    settings: AppSettings,
    auth_service: AuthService,
    user_service: UserService,
    oauth_error: Optional[str] = None,
) -> None:
    """Render the clean SaaS Login / Registration gate when no valid backend session is active."""
    _, center_col, _ = st.columns([1.15, 1.7, 1.15])
    with center_col:
        render_html(
            f'<div style="text-align:center;margin:18px 0 18px;">'
            f'<div style="display:inline-flex;align-items:center;gap:10px;">{BRAND_SVG}'
            f'<span style="font-size:24px;font-weight:700;letter-spacing:-0.02em;color:#172033;">QnAGPT</span></div>'
            f'<div style="font-size:13.5px;color:#64748B;margin-top:6px;">'
            f"Sign in to access your isolated Knowledge Base, conversations, and MCP workspace."
            f"</div></div>"
        )

        if oauth_error:
            st.error(oauth_error)

        with st.container(border=True):
            tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

            with tab_login:
                with st.form("auth_login_form", clear_on_submit=False):
                    email = st.text_input("Email", placeholder="you@company.com", key="auth_login_email")
                    password = st.text_input("Password", type="password", key="auth_login_pw")
                    submitted = st.form_submit_button(
                        "Sign In",
                        type="primary",
                        use_container_width=True,
                    )
                if submitted:
                    try:
                        res = auth_service.login_with_password(email=email, password=password)
                        _apply_auth_result(res, user_service)
                        st.rerun()
                    except (AuthenticationError, SecurityError) as exc:
                        st.error(str(exc))

                st.divider()
                st.caption("Single Sign-On (OAuth2)")
                g_col, gh_col = st.columns(2)
                with g_col:
                    if auth_service.is_oauth_configured("google"):
                        if st.button("Continue with Google", use_container_width=True, key="btn_oauth_google"):
                            try:
                                start_res = auth_service.start_oauth_login("google")
                                st.link_button(
                                    "Open Google Consent",
                                    start_res.authorization_url,
                                    type="primary",
                                    use_container_width=True,
                                )
                            except ConfigurationError as exc:
                                st.info(str(exc))
                    else:
                        st.button(
                            "Google OAuth",
                            disabled=True,
                            use_container_width=True,
                            key="btn_oauth_google_off",
                            help="Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env to enable Google Sign-In.",
                        )
                with gh_col:
                    if auth_service.is_oauth_configured("github"):
                        if st.button("Continue with GitHub", use_container_width=True, key="btn_oauth_github"):
                            try:
                                start_res = auth_service.start_oauth_login("github")
                                st.link_button(
                                    "Open GitHub Consent",
                                    start_res.authorization_url,
                                    type="primary",
                                    use_container_width=True,
                                )
                            except ConfigurationError as exc:
                                st.info(str(exc))
                    else:
                        st.button(
                            "GitHub OAuth",
                            disabled=True,
                            use_container_width=True,
                            key="btn_oauth_github_off",
                            help="Add GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET to .env to enable GitHub Sign-In.",
                        )

                if not (auth_service.is_oauth_configured("google") or auth_service.is_oauth_configured("github")):
                    st.caption(
                        "OAuth sign-in activates automatically once `GOOGLE_CLIENT_ID`/`SECRET` or "
                        "`GITHUB_CLIENT_ID`/`SECRET` are configured in `.env`."
                    )

            with tab_reg:
                with st.form("auth_register_form", clear_on_submit=False):
                    reg_name = st.text_input("Full Name", placeholder="Alex Morgan", key="auth_reg_name")
                    reg_email = st.text_input("Work or Personal Email", placeholder="alex@company.com", key="auth_reg_email")
                    reg_pw = st.text_input(
                        "Password",
                        type="password",
                        key="auth_reg_pw",
                        help="Minimum 8 characters with uppercase, lowercase, and a number or symbol.",
                    )
                    reg_pw2 = st.text_input("Confirm Password", type="password", key="auth_reg_pw2")
                    reg_btn = st.form_submit_button(
                        "Create Account",
                        type="primary",
                        use_container_width=True,
                    )
                if reg_btn:
                    if reg_pw != reg_pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            res = auth_service.register_with_password(
                                email=reg_email,
                                password=reg_pw,
                                full_name=reg_name,
                            )
                            _apply_auth_result(res, user_service)
                            st.rerun()
                        except AuthenticationError as exc:
                            st.error(str(exc))


def render_account_page(
    settings: AppSettings,
    auth_session: AuthenticatedSession,
    auth_service: AuthService,
    user_service: UserService,
    vector_store: Optional[QdrantVectorStoreManager] = None,
    on_signed_out: Optional[callable] = None,  # type: ignore[valid-type]
) -> None:
    """Render authenticated user profile, password security, legacy claim, and account deletion."""
    user = auth_session.user
    render_page_header(
        "Account & Security",
        f"Signed in as {user.email} ({user.id})",
    )

    render_metric_strip(
        [
            ("Account Status", "Active" if user.is_active else "Disabled"),
            ("Email Verified", "Yes" if user.is_verified else "Unverified"),
            ("Password Login", "Configured" if user.has_password else "OAuth Only"),
            ("Session Expires", auth_session.expires_at.strftime("%Y-%m-%d %H:%M UTC")),
        ]
    )

    if auth_service.has_unclaimed_legacy_data():
        render_section_header("Legacy Workspace Data Available")
        with st.container(border=True):
            st.info(
                "Pre-migration conversations, NeuralVerge MCP server configuration, and local documents from "
                "the previous single-user workspace can be imported safely into your account."
            )
            if st.button("Claim Legacy Workspace Data", type="primary", key="acct_claim_legacy"):
                token = str(st.session_state.get("auth_session_token") or "")
                moved = auth_service.claim_legacy_workspace_for_session(token)
                faiss_ok = {"documents_migrated": 0, "chunks_migrated": 0}
                if vector_store is not None and (Path(settings.vector_store_dir) / "local_user").is_dir():
                    try:
                        faiss_ok = migrate_faiss_user_to_qdrant(
                            faiss_dir=settings.vector_store_dir,
                            qdrant_store=vector_store,
                            source_user_id="local_user",
                            target_user_id=user.id,
                        )
                    except Exception as exc:
                        st.warning(f"Database claimed cleanly, but FAISS -> Qdrant vector transfer deferred: {exc}")
                st.success(
                    f"Imported {moved.get('conversations', 0)} conversations, "
                    f"{moved.get('mcp_servers', 0)} MCP server(s), and "
                    f"{faiss_ok.get('documents_migrated', 0)} vector document(s)."
                )
                st.rerun()

    tab_prof, tab_sec, tab_audit = st.tabs(["Profile & Sessions", "Password & Linked Logins", "Security Audit Log"])

    with tab_prof:
        render_section_header("Profile Details")
        with st.form("acct_profile_form"):
            new_name = st.text_input("Full Name", value=user.full_name, key="acct_full_name")
            save_prof = st.form_submit_button("Save Profile", type="primary")
        if save_prof:
            user_service.update_profile(user_id=user.id, full_name=new_name)
            st.success("Profile updated.")
            st.rerun()

        st.divider()
        c_out, _ = st.columns([1.4, 3])
        with c_out:
            if st.button("Sign Out", icon=":material/logout:", use_container_width=True, key="acct_sign_out"):
                auth_service.logout(st.session_state.get("auth_session_token"))
                if on_signed_out:
                    on_signed_out()
                st.rerun()

    with tab_sec:
        render_section_header("Change Password")
        with st.form("acct_pw_form"):
            cur_pw = st.text_input("Current Password", type="password", key="acct_cur_pw") if user.has_password else ""
            new_pw = st.text_input("New Password", type="password", key="acct_new_pw")
            conf_pw = st.text_input("Confirm New Password", type="password", key="acct_conf_pw")
            pw_submit = st.form_submit_button("Update Password", type="primary")
        if pw_submit:
            if new_pw != conf_pw:
                st.error("New passwords do not match.")
            else:
                try:
                    user_service.change_password(
                        user_id=user.id,
                        current_password=cur_pw,
                        new_password=new_pw,
                    )
                    st.success("Password updated. Other active sessions were revoked—please sign in again.")
                    if on_signed_out:
                        on_signed_out()
                    st.rerun()
                except AuthenticationError as exc:
                    st.error(str(exc))

        render_section_header("Linked OAuth Providers")
        linked = user_service.list_connected_providers(user.id)
        if linked:
            for item in linked:
                render_html(
                    f'<div style="padding:8px 10px;border:1px solid #E2E8F0;border-radius:8px;margin-bottom:6px;">'
                    f'<strong>{html.escape(item["provider"].title())}</strong> · {html.escape(item["email"])} '
                    f'{status_pill_html("Verified" if item["verified"] == "Yes" else "Unverified", "ok" if item["verified"] == "Yes" else "warn")}'
                    f"</div>"
                )
        else:
            st.caption("No external Google or GitHub identities linked yet.")

        st.divider()
        render_section_header("Danger Zone: Delete Account")
        with st.expander("Permanently Delete Account & Workspace Data"):
            st.warning(
                "This permanently deletes your conversations, uploaded files, Qdrant vectors, MCP configs, and sessions."
            )
            del_pw = st.text_input("Confirm Current Password", type="password", key="acct_del_pw") if user.has_password else ""
            del_confirm = st.checkbox("I understand this cannot be undone", key="acct_del_chk")
            if st.button("Delete My Account", type="primary", disabled=not del_confirm, key="acct_del_btn"):
                try:
                    user_service.delete_account(user_id=user.id, confirm_password=del_pw)
                    if on_signed_out:
                        on_signed_out()
                    st.rerun()
                except AuthenticationError as exc:
                    st.error(str(exc))

    with tab_audit:
        render_section_header("Recent Account Activity")
        logs = user_service.list_audit_logs(user.id, limit=30)
        if not logs:
            st.caption("No security audit events recorded yet.")
        else:
            for row in logs:
                st.caption(
                    f"`{row['created_at']}` — **{row['action']}** ({row['resource_type']}: `{row['resource_id']}`)"
                )
