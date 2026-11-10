"""Initial QnAGPT multi-user SQLite platform schema.

Revision ID: 0001_initial_platform
Revises:
Create Date: 2026-11-10 23:10:00
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

import qna_bot.models  # noqa: F401
from qna_bot.models.base import Base

revision: str = "0001_initial_platform"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create any missing tables cleanly across fresh or pre-existing SQLite databases."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    """Drop normalized platform tables while keeping raw legacy checkpoints untouched unless full reset."""
    bind = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        table.drop(bind=bind, checkfirst=True)
