"""User and per-user preference repository with strict transactional safety."""

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
import uuid

from sqlalchemy import delete, func, select

from qna_bot.core.exceptions import AuthorizationError
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.user import User, UserSetting


class UserRepository:
    """Manages user profiles, password hashes, and normalized user_settings."""

    def __init__(self, db: SQLAlchemyDatabase) -> None:
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self.db.session_scope() as session:
            return session.get(User, user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        norm = (email or "").strip().lower()
        with self.db.session_scope() as session:
            stmt = select(User).where(func.lower(User.email) == norm)
            return session.scalar(stmt)

    def count_real_users(self) -> int:
        with self.db.session_scope() as session:
            stmt = select(func.count()).select_from(User).where(User.id != "local_user")
            return int(session.scalar(stmt) or 0)

    def create_user(
        self,
        email: str,
        password_hash: Optional[str] = None,
        full_name: str = "",
        avatar_url: str = "",
        is_verified: bool = False,
        user_id: Optional[str] = None,
    ) -> User:
        norm = (email or "").strip().lower()
        uid = user_id or f"usr_{uuid.uuid4().hex[:16]}"
        with self.db.session_scope() as session:
            user = User(
                id=uid,
                email=norm,
                password_hash=password_hash,
                full_name=full_name.strip(),
                avatar_url=avatar_url.strip(),
                is_active=True,
                is_verified=bool(is_verified),
            )
            session.add(user)
            session.flush()
            session.refresh(user)
            return user

    def update_profile(
        self,
        user_id: str,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        with self.db.session_scope() as session:
            user = session.get(User, user_id)
            if user is None:
                raise AuthorizationError("Authenticated user account was not found.")
            if full_name is not None:
                user.full_name = full_name.strip()[:160]
            if avatar_url is not None:
                user.avatar_url = avatar_url.strip()[:512]
            user.updated_at = datetime.now(timezone.utc)
            session.flush()
            session.refresh(user)
            return user

    def update_password_hash(self, user_id: str, password_hash: str) -> None:
        with self.db.session_scope() as session:
            user = session.get(User, user_id)
            if user is None:
                raise AuthorizationError("Authenticated user account was not found.")
            user.password_hash = password_hash
            user.updated_at = datetime.now(timezone.utc)

    def mark_login(self, user_id: str, mark_verified: bool = False) -> None:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            user = session.get(User, user_id)
            if user is not None:
                user.last_login_at = now
                if mark_verified:
                    user.is_verified = True
                user.updated_at = now

    def get_settings(self, user_id: str) -> Dict[str, str]:
        with self.db.session_scope() as session:
            rows = session.scalars(
                select(UserSetting).where(UserSetting.user_id == user_id)
            ).all()
            return {row.setting_key: row.setting_value for row in rows}

    def upsert_settings(self, user_id: str, updates: Mapping[str, Any]) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            for key, raw_val in updates.items():
                k = str(key).strip()[:128]
                if not k:
                    continue
                v = str(raw_val)
                existing = session.scalar(
                    select(UserSetting).where(
                        UserSetting.user_id == user_id,
                        UserSetting.setting_key == k,
                    )
                )
                if existing is None:
                    session.add(
                        UserSetting(user_id=user_id, setting_key=k, setting_value=v, updated_at=now)
                    )
                else:
                    existing.setting_value = v
                    existing.updated_at = now
        return self.get_settings(user_id)

    def delete_user(self, user_id: str) -> bool:
        with self.db.session_scope() as session:
            res = session.execute(delete(User).where(User.id == user_id))
            return int(res.rowcount or 0) > 0
