"""Backend-enforced human-in-the-loop approval management for MCP tool execution."""

import hashlib
import json
from typing import Any, List, Mapping, Optional
import uuid

from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import ApprovalRecord
from qna_bot.database.repositories import ApprovalRepository
from qna_bot.mcp.security import format_masked_arguments


def compute_arguments_hash(arguments: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash over canonical JSON tool arguments."""
    canonical = json.dumps(dict(arguments or {}), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ApprovalManager:
    """Manages persistent approvals strictly bound to user, thread, server, tool, and arguments."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()
        self.repo = ApprovalRepository(self.db)

    def create_pending_approval(
        self,
        user_id: str,
        thread_id: str,
        server_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        category: str,
        reason: str,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=f"appr_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            thread_id=thread_id,
            server_id=server_id,
            tool_name=tool_name,
            arguments_hash=compute_arguments_hash(arguments),
            masked_arguments=format_masked_arguments(arguments),
            category=category,
            reason=reason,
            status="PENDING",
        )
        return self.repo.create(record)

    def decide(
        self,
        approval_id: str,
        user_id: str,
        approved: bool,
    ) -> Optional[ApprovalRecord]:
        status = "APPROVED" if approved else "REJECTED"
        return self.repo.decide(approval_id=approval_id, user_id=user_id, status=status)

    def verify_and_consume(
        self,
        approval_id: str,
        user_id: str,
        thread_id: str,
        server_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> bool:
        """Verify an exact backend approval match before allowing execution."""
        record = self.repo.get(approval_id=approval_id, user_id=user_id)
        if record is None:
            return False
        if record.status != "APPROVED":
            return False
        if record.thread_id != thread_id or record.server_id != server_id or record.tool_name != tool_name:
            return False
        if record.arguments_hash != compute_arguments_hash(arguments):
            return False

        self.repo.decide(approval_id=approval_id, user_id=user_id, status="CONSUMED")
        return True

    def list_pending(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> List[ApprovalRecord]:
        return self.repo.list_pending(user_id=user_id, thread_id=thread_id)
