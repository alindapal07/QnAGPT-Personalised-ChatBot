"""Authentication identities, persistent sessions, and OAuth pending CSRF states repository."""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid

from sqlalchemy import delete, select, update

from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.user import AuthIdentity, OAuthPendingState, User, UserSession


class AuthRepository:
    """Manages OAuth identity links, single-use CSRF states, and session token lookups."""

    def __init__(self, db: SQLAlchemyDatabase) -> None:
        self.db = db

    def get_identity(self, provider: str, provider_user_id: str) -> Optional[AuthIdentity]:
        with self.db.session_scope() as session:
            stmt = select(AuthIdentity).where(
                AuthIdentity.provider == provider.lower(),
                AuthIdentity.provider_user_id == str(provider_user_id),
            )
            return session.scalar(stmt)

    def list_identities(self, user_id: str) -> List[AuthIdentity]:
        with self.db.session_scope() as session:
            return list(
                session.scalars(
                    select(AuthIdentity)
                    .where(AuthIdentity.user_id == user_id)
                    .order_by(AuthIdentity.created_at)
                ).all()
            )

    def link_identity(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: str,
        email_verified: bool,
    ) -> AuthIdentity:
        prov = provider.lower()
        sub = str(provider_user_id)
        with self.db.session_scope() as session:
            existing = session.scalar(
                select(AuthIdentity).where(
                    AuthIdentity.provider == prov,
                    AuthIdentity.provider_user_id == sub,
                )
            )
            if existing is not None:
                existing.provider_email = provider_email.strip().lower()
                existing.email_verified = bool(email_verified)
                session.flush()
                session.refresh(existing)
                return existing

            identity = AuthIdentity(
                id=f"aid_{uuid.uuid4().hex[:16]}",
                user_id=user_id,
                provider=prov,
                provider_user_id=sub,
                provider_email=provider_email.strip().lower(),
                email_verified=bool(email_verified),
            )
            session.add(identity)
            session.flush()
            session.refresh(identity)
            return identity

    def create_session(
        self,
        user_id: str,
        token_hash: str,
        csrf_token: str,
        expires_at: datetime,
    ) -> UserSession:
        with self.db.session_scope() as session:
            sess = UserSession(
                id=f"ses_{uuid.uuid4().hex[:16]}",
                user_id=user_id,
                token_hash=token_hash,
                csrf_token=csrf_token,
                expires_at=expires_at,
            )
            session.add(sess)
            session.flush()
            session.refresh(sess)
            return sess

    def get_active_session_with_user(
        self,
        token_hash: str,
    ) -> Optional[Tuple[UserSession, User]]:
        if not token_hash:
            return None
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            stmt = (
                select(UserSession, User)
                .join(User, User.id == UserSession.user_id)
                .where(
                    UserSession.token_hash == token_hash,
                    UserSession.revoked_at.is_(None),
                    User.is_active.is_(True),
                )
            )
            row = session.execute(stmt).first()
            if row is None:
                return None
            user_session, user = row[0], row[1]
            exp = user_session.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= now:
                return None
            return user_session, user

    def revoke_session_by_hash(self, token_hash: str) -> bool:
        if not token_hash:
            return False
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            res = session.execute(
                update(UserSession)
                .where(UserSession.token_hash == token_hash, UserSession.revoked_at.is_(None))
                .values(revoked_at=now)
            )
            return int(res.rowcount or 0) > 0

    def revoke_user_sessions(
        self,
        user_id: str,
        except_token_hash: Optional[str] = None,
    ) -> int:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            stmt = update(UserSession).where(
                UserSession.user_id == user_id,
                UserSession.revoked_at.is_(None),
            )
            if except_token_hash:
                stmt = stmt.where(UserSession.token_hash != except_token_hash)
            res = session.execute(stmt.values(revoked_at=now))
            return int(res.rowcount or 0)

    def create_oauth_state(
        self,
        state: str,
        provider: str,
        code_verifier: str,
        nonce: str,
        redirect_uri: str,
        expires_at: datetime,
    ) -> OAuthPendingState:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            # Opportunistic expired state cleanup
            session.execute(delete(OAuthPendingState).where(OAuthPendingState.expires_at <= now))
            rec = OAuthPendingState(
                state=state,
                provider=provider.lower(),
                code_verifier=code_verifier,
                nonce=nonce,
                redirect_uri=redirect_uri,
                expires_at=expires_at,
            )
            session.add(rec)
            session.flush()
            session.refresh(rec)
            return rec

    def consume_oauth_state(
        self,
        provider: str,
        state: str,
    ) -> Optional[OAuthPendingState]:
        if not state:
            return None
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            rec = session.scalar(
                select(OAuthPendingState).where(
                    OAuthPendingState.state == state,
                    OAuthPendingState.provider == provider.lower(),
                    OAuthPendingState.consumed_at.is_(None),
                )
            )
            if rec is None:
                return None
            exp = rec.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= now:
                return None
            rec.consumed_at = now
            session.flush()
            session.refresh(rec)
            return rec
