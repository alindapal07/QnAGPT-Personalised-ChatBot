"""User profile, preferences, password management, and account deletion service."""

from typing import Any, Dict, List, Mapping, Optional

from qna_bot.core.exceptions import AuthenticationError
from qna_bot.core.security import hash_password, validate_password_strength, verify_password
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.repositories import AuditAndTelemetryRepository, AuthRepository, UserRepository
from qna_bot.schemas.auth import UserProfile
from qna_bot.storage.base import StorageProvider


class UserService:
    """Backend service managing user profiles, preferences, sessions, and clean account deletion."""

    def __init__(
        self,
        db: SQLAlchemyDatabase,
        storage: Optional[StorageProvider] = None,
        vector_store: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.storage = storage
        self.vector_store = vector_store
        self.users = UserRepository(db)
        self.auth = AuthRepository(db)
        self.audit = AuditAndTelemetryRepository(db)

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        user = self.users.get_by_id(user_id)
        if user is None:
            return None
        return UserProfile(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            is_active=user.is_active,
            is_verified=user.is_verified,
            has_password=bool(user.password_hash),
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )

    def update_profile(
        self,
        user_id: str,
        full_name: str,
        avatar_url: Optional[str] = None,
    ) -> UserProfile:
        updated = self.users.update_profile(user_id=user_id, full_name=full_name, avatar_url=avatar_url)
        self.audit.log_audit(
            user_id=user_id,
            action="user.update_profile",
            resource_type="user",
            resource_id=user_id,
        )
        return UserProfile(
            id=updated.id,
            email=updated.email,
            full_name=updated.full_name,
            avatar_url=updated.avatar_url,
            is_active=updated.is_active,
            is_verified=updated.is_verified,
            has_password=bool(updated.password_hash),
            created_at=updated.created_at,
            last_login_at=updated.last_login_at,
        )

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> None:
        user = self.users.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("User account not found.")
        if user.password_hash:
            if not verify_password(current_password, user.password_hash):
                raise AuthenticationError("Current password is incorrect.")
        validate_password_strength(new_password)
        self.users.update_password_hash(user_id, hash_password(new_password))
        self.auth.revoke_user_sessions(user_id)
        self.audit.log_audit(
            user_id=user_id,
            action="user.change_password",
            resource_type="user",
            resource_id=user_id,
        )

    def get_settings(self, user_id: str) -> Dict[str, str]:
        return self.users.get_settings(user_id)

    def save_settings(self, user_id: str, updates: Mapping[str, Any]) -> Dict[str, str]:
        res = self.users.upsert_settings(user_id, updates)
        self.audit.log_audit(
            user_id=user_id,
            action="user.update_settings",
            resource_type="user_settings",
            resource_id=user_id,
            metadata={"keys": sorted(str(k) for k in updates.keys())},
        )
        return res

    def list_connected_providers(self, user_id: str) -> List[Dict[str, str]]:
        return [
            {
                "provider": item.provider,
                "email": item.provider_email,
                "verified": "Yes" if item.email_verified else "No",
            }
            for item in self.auth.list_identities(user_id)
        ]

    def list_audit_logs(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        return [
            {
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "created_at": log.created_at.isoformat() if log.created_at else "",
            }
            for log in self.audit.list_audit_logs(user_id, limit=limit)
        ]

    def delete_account(self, user_id: str, confirm_password: str = "") -> bool:
        """Verify ownership, purge Qdrant vectors & local storage, and cascade-delete user SQLite data."""
        user = self.users.get_by_id(user_id)
        if user is None:
            return False
        if user.password_hash and confirm_password:
            if not verify_password(confirm_password, user.password_hash):
                raise AuthenticationError("Password confirmation failed.")
        elif user.password_hash and not confirm_password:
            raise AuthenticationError("Current password is required to delete your account.")

        self.auth.revoke_user_sessions(user_id)
        if self.vector_store is not None:
            try:
                self.vector_store.clear(user_id)
            except Exception:
                pass
        if self.storage is not None:
            try:
                self.storage.delete_user_storage(user_id)
            except Exception:
                pass
        self.audit.log_audit(
            user_id=None,
            action="user.delete_account",
            resource_type="user",
            resource_id=user_id,
        )
        return self.users.delete_user(user_id)
