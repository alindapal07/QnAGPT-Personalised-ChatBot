"""Pydantic v2 DTOs for authentication, OAuth login, and user profiles."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Public-safe authenticated user profile (never includes password hashes)."""

    id: str
    email: str
    full_name: str = ""
    avatar_url: str = ""
    is_active: bool = True
    is_verified: bool = False
    has_password: bool = False
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class AuthenticatedSession(BaseModel):
    """Verified server-side session resolved from token_hash lookup."""

    session_id: str
    user: UserProfile
    csrf_token: str
    expires_at: datetime


class AuthResult(BaseModel):
    """Result returned on successful registration, password login, or OAuth callback."""

    session_token: str
    session: AuthenticatedSession
    is_new_user: bool = False


class OAuthStartResult(BaseModel):
    """Server-generated authorization URL and persisted CSRF state."""

    provider: str
    authorization_url: str
    state: str
    redirect_uri: str = Field(...)
