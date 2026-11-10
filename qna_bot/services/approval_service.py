"""Service layer for managing pending human-in-the-loop tool approvals."""

from typing import List, Optional

from qna_bot.agent.approval import ApprovalManager
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import ApprovalRecord


class ApprovalService:
    """Backend service coordinating user-scoped tool approval decisions."""

    def __init__(
        self,
        approval_manager: Optional[ApprovalManager] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.manager = approval_manager or ApprovalManager(db_manager=self.db)

    def list_pending(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> List[ApprovalRecord]:
        return self.manager.list_pending(user_id=user_id, thread_id=thread_id)

    def record_decision(
        self,
        approval_id: str,
        user_id: str,
        approved: bool,
    ) -> Optional[ApprovalRecord]:
        return self.manager.decide(
            approval_id=approval_id,
            user_id=user_id,
            approved=approved,
        )
