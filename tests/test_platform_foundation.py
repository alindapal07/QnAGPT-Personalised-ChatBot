"""End-to-end tests for SQLAlchemy 2.x + Alembic, Argon2id/OAuth2 auth, LocalStorage, and Qdrant isolation."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from langchain_core.documents import Document
import pytest
from sqlalchemy import text

from qna_bot.config.settings import AppSettings
from qna_bot.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    SecurityError,
)
from qna_bot.database.legacy_migration import claim_legacy_workspace, ensure_legacy_workspace_synced
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.rag.qdrant_store import QdrantVectorStoreManager, migrate_faiss_user_to_qdrant
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.repositories import ConversationRepository, UserRepository
from qna_bot.services.auth_service import AuthService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.user_service import UserService
from qna_bot.storage.local import LocalStorageProvider
from tests.test_rag import DeterministicTestEmbeddings


def _make_settings(tmp_path: Path, **overrides) -> AppSettings:
    defaults = {
        "groq_api_key": "gsk_test",
        "data_dir": tmp_path / "data",
        "storage_dir": tmp_path / "storage",
        "vector_store_dir": tmp_path / "data" / "vector_store",
        "sqlite_db_path": tmp_path / "data" / "app.db",
        "langgraph_checkpoint_db": tmp_path / "data" / "app.db",
        "session_secret_key": "unit-test-session-secret",
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


def test_alembic_upgrade_wal_foreign_keys_and_legacy_migration(tmp_path: Path) -> None:
    db_file = tmp_path / "fresh.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_file.as_posix()}")
    command.upgrade(cfg, "head")

    db = SQLAlchemyDatabase(db_file, auto_create=False)
    try:
        with db.engine.connect() as conn:
            fk = conn.execute(text("PRAGMA foreign_keys;")).scalar()
            jm = str(conn.execute(text("PRAGMA journal_mode;")).scalar() or "").lower()
        assert fk == 1
        assert jm == "wal"
    finally:
        db.close()

    # Verify non-destructive upgrade against existing legacy schema
    live_copy = tmp_path / "legacy_copy.db"
    src_live = Path("data/chatbot.db")
    if src_live.is_file():
        shutil.copy2(src_live, live_copy)
        cfg2 = Config("alembic.ini")
        cfg2.set_main_option("sqlalchemy.url", f"sqlite:///{live_copy.as_posix()}")
        command.upgrade(cfg2, "head")
        db2 = SQLAlchemyDatabase(live_copy, auto_create=False)
        try:
            sync = ensure_legacy_workspace_synced(db2)
            assert sync["threads_migrated"] >= 1
            owner = UserRepository(db2).create_user("owner@company.com", full_name="Owner", is_verified=True)
            claimed = claim_legacy_workspace(db2, owner.id)
            assert claimed.get("conversations", 0) >= 1
            assert len(ConversationRepository(db2).list_for_user(owner.id)) >= 1
        finally:
            db2.close()


def test_password_auth_rate_limiting_sessions_and_account_deletion(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    db = SQLAlchemyDatabase(settings.resolved_database_url)
    storage = LocalStorageProvider(settings.storage_dir)
    qdrant = QdrantVectorStoreManager(
        settings=settings,
        location=":memory:",
        embedding_factory=DeterministicTestEmbeddings,
    )
    auth = AuthService(db=db, settings=settings)
    user_svc = UserService(db=db, storage=storage, vector_store=qdrant)
    try:
        with pytest.raises(AuthenticationError):
            auth.register_with_password("bad-email", "StrongPass1!")
        with pytest.raises(AuthenticationError):
            auth.register_with_password("alice@company.com", "weak")

        reg = auth.register_with_password("Alice@Company.com", "StrongPass1!", full_name="Alice A")
        assert reg.is_new_user
        alice_id = reg.session.user.id
        stored_user = UserRepository(db).get_by_id(alice_id)
        assert stored_user is not None
        assert stored_user.password_hash is not None
        assert stored_user.password_hash.startswith("$argon2id$")

        # Verify session resolution & logout revocation
        resolved = auth.validate_session(reg.session_token)
        assert resolved is not None and resolved.user.id == alice_id
        assert auth.logout(reg.session_token) is True
        assert auth.validate_session(reg.session_token) is None

        # Wrong password triggers uniform error and rate limit after 5 attempts
        for _ in range(5):
            with pytest.raises(AuthenticationError, match="Invalid email or password"):
                auth.login_with_password("alice@company.com", "WrongPass9!", client_key="127.0.0.1")
        with pytest.raises(SecurityError, match="Too many failed"):
            auth.login_with_password("alice@company.com", "StrongPass1!", client_key="127.0.0.1")

        # Another client succeeds after password login & supports profile/password/delete
        login_res = auth.login_with_password("alice@company.com", "StrongPass1!", client_key="other")
        user_svc.update_profile(alice_id, "Alice Updated")
        assert user_svc.get_profile(alice_id).full_name == "Alice Updated"
        user_svc.save_settings(alice_id, {"text_model": "openai/gpt-oss-120b"})
        assert user_svc.get_settings(alice_id)["text_model"] == "openai/gpt-oss-120b"

        storage.save_file(alice_id, "documents", "secret.txt", b"hello")
        qdrant.add_documents(
            [Document(page_content="Confidential Alice strategy notes.", metadata={"chunk_index": 0})],
            user_id=alice_id,
            document_meta={"document_id": "doc_a1", "source": "secret.txt", "file_hash": "h1"},
        )
        assert len(qdrant.search_documents("strategy", user_id=alice_id)) == 1

        user_svc.change_password(alice_id, "StrongPass1!", "NewStrongPass2@")
        assert auth.validate_session(login_res.session_token) is None

        assert user_svc.delete_account(alice_id, confirm_password="NewStrongPass2@") is True
        assert UserRepository(db).get_by_id(alice_id) is None
        assert storage.list_files(alice_id, "documents") == []
        assert qdrant.list_documents(alice_id) == []
    finally:
        db.close()


def test_oauth_config_check_csrf_state_and_verified_email_linking(tmp_path: Path) -> None:
    settings_unconfigured = _make_settings(tmp_path)
    db = SQLAlchemyDatabase(settings_unconfigured.resolved_database_url)
    auth_no_cfg = AuthService(db=db, settings=settings_unconfigured)
    try:
        assert not auth_no_cfg.is_oauth_configured("google")
        with pytest.raises(ConfigurationError, match="GOOGLE_CLIENT_ID"):
            auth_no_cfg.start_oauth_login("google")

        class _FakeOAuthClient:
            def __init__(self, **_kwargs) -> None:
                pass

            def create_authorization_url(self, url: str, **kwargs):
                return f"{url}?state={kwargs['state']}", kwargs["state"]

            def fetch_token(self, **_kwargs):
                return {"access_token": "tok_123"}

            def get(self, url: str):
                if "userinfo" in url:
                    return SimpleNamespace(
                        status_code=200,
                        raise_for_status=lambda: None,
                        json=lambda: {
                            "sub": "google-sub-42",
                            "email": "bob@company.com",
                            "email_verified": False,
                            "name": "Bob Unverified",
                        },
                    )
                raise AssertionError(f"Unexpected URL {url}")

        cfg_settings = _make_settings(
            tmp_path,
            google_client_id="gid",
            google_client_secret="gsec",
        )
        auth = AuthService(db=db, settings=cfg_settings, oauth_client_factory=_FakeOAuthClient)
        # Seed existing password account with bob@company.com
        auth.register_with_password("bob@company.com", "StrongPass1!", full_name="Bob Local")

        started = auth.start_oauth_login("google")
        assert started.state in started.authorization_url

        # Unverified OAuth email must NOT link into existing password account
        with pytest.raises(AuthenticationError, match="unverified"):
            auth.complete_oauth_login("google", code="code_abc", state=started.state)

        # Replay of consumed state must fail immediately
        with pytest.raises(SecurityError, match="CSRF"):
            auth.complete_oauth_login("google", code="code_abc", state=started.state)
    finally:
        db.close()


def test_local_storage_blocks_path_traversal_and_cross_user_access(tmp_path: Path) -> None:
    storage = LocalStorageProvider(tmp_path / "storage")
    with pytest.raises(SecurityError):
        storage.save_file("usr_alice", "documents", "../../etc/passwd.txt", b"safe-content")

    meta = storage.save_file("usr_alice", "documents", "report.txt", b"safe-content")
    assert meta.filename == "report.txt"
    assert storage.read_file("usr_alice", meta.relative_path) == b"safe-content"
    assert meta.checksum == hashlib.sha256(b"safe-content").hexdigest()

    for bad_path in ("../usr_alice/documents/x", "/etc/passwd", "C:\\Windows\\win.ini", "..\\evil.txt"):
        with pytest.raises(SecurityError):
            storage.read_file("usr_bob", bad_path)

    with pytest.raises(SecurityError):
        storage.save_file("../escape", "documents", "ok.txt", b"x")


def test_qdrant_isolation_durable_retry_and_faiss_migration(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    db = SQLAlchemyDatabase(settings.resolved_database_url)
    storage = LocalStorageProvider(settings.storage_dir)
    qdrant = QdrantVectorStoreManager(
        settings=settings,
        location=":memory:",
        embedding_factory=DeterministicTestEmbeddings,
    )
    auth = AuthService(db=db, settings=settings)
    alice_id = auth.register_with_password("alice@iso.io", "StrongPass1!").session.user.id
    bob_id = auth.register_with_password("bob@iso.io", "StrongPass1!").session.user.id

    try:
        # Failure during Qdrant upsert must still preserve raw file + mark FAILED for retry
        broken_qdrant = QdrantVectorStoreManager(
            settings=settings,
            location=":memory:",
            embedding_factory=lambda: (_ for _ in ()).throw(RuntimeError("Qdrant down")),
        )
        failing_doc_svc = DocumentService(
            settings=settings,
            vector_store=broken_qdrant,
            storage=storage,
            db=db,
        )
        res_fail = failing_doc_svc.ingest_files([("notes.md", b"# Architecture\nQdrant isolation notes.")], user_id=alice_id)
        assert res_fail[0].status == "failed"
        listed = failing_doc_svc.list_documents(alice_id)
        assert len(listed) == 1
        assert listed[0]["processing_status"] == "failed"
        doc_id = listed[0]["document_id"]
        assert failing_doc_svc.read_original_file(doc_id, alice_id) == ("notes.md", b"# Architecture\nQdrant isolation notes.")

        # Recovery using real Qdrant retry without re-uploading
        working_doc_svc = DocumentService(
            settings=settings,
            vector_store=qdrant,
            storage=storage,
            db=db,
        )
        retried = working_doc_svc.retry_document(doc_id, alice_id)
        assert retried.status == "ingested"
        assert working_doc_svc.list_documents(alice_id)[0]["processing_status"] == "indexed"

        # Tenant isolation: Bob gets zero docs, cannot read/delete Alice's doc or conversation
        assert working_doc_svc.list_documents(bob_id) == []
        assert working_doc_svc.read_original_file(doc_id, bob_id) is None
        assert working_doc_svc.remove_document(doc_id, bob_id) is False
        assert qdrant.search_documents("Qdrant isolation", user_id=alice_id)
        assert qdrant.search_documents("Qdrant isolation", user_id=bob_id) == []

        conv_repo = ConversationRepository(db)
        conv = conv_repo.create(alice_id, title="Alice Private Thread")
        msg = conv_repo.append_message(conv.id, alice_id, "assistant", "Secret reply")
        with pytest.raises(AuthorizationError):
            conv_repo.append_message(conv.id, bob_id, "user", "Hijack")
        with pytest.raises(AuthorizationError):
            conv_repo.record_feedback(bob_id, msg.id, "down")
        assert conv_repo.delete(conv.id, bob_id) is False

        # Legacy FAISS -> Qdrant migration preserves document cleanly
        faiss_mgr = FAISSVectorStoreManager(
            base_dir=settings.vector_store_dir,
            embeddings=DeterministicTestEmbeddings(),
        )
        faiss_mgr.add_documents(
            [Document(page_content="Payslip net salary 95000 INR.", metadata={"source": "Payslip_9_2026.pdf", "chunk_index": 0})],
            user_id="local_user",
            document_meta={"document_id": "legacy_pay_1", "filename": "Payslip_9_2026.pdf", "file_hash": "pay_hash_1"},
        )
        mig = migrate_faiss_user_to_qdrant(
            faiss_dir=settings.vector_store_dir,
            qdrant_store=qdrant,
            source_user_id="local_user",
            target_user_id=bob_id,
        )
        assert mig == {"documents_migrated": 1, "chunks_migrated": 1}
        bob_hits = qdrant.search_documents("Payslip salary", user_id=bob_id)
        assert len(bob_hits) == 1 and bob_hits[0].source == "Payslip_9_2026.pdf"
    finally:
        db.close()
