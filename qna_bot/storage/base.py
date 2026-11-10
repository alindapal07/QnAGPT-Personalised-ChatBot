"""Abstract storage provider interface decoupled from local filesystem details."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

StorageBucket = Literal["documents", "images", "attachments", "exports"]
STORAGE_BUCKETS = ("documents", "images", "attachments", "exports")


@dataclass(frozen=True)
class StoredFileMeta:
    """Metadata returned when a user file is stored durably."""

    relative_path: str
    absolute_path: str
    filename: str
    bucket: str
    size_bytes: int
    checksum: str


class StorageProvider(ABC):
    """Abstract storage contract supporting local filesystem now or cloud object stores later."""

    @abstractmethod
    def save_file(
        self,
        user_id: str,
        bucket: StorageBucket,
        filename: str,
        data: bytes,
    ) -> StoredFileMeta:
        """Persist bytes safely within the authenticated user's isolated storage root."""

    @abstractmethod
    def read_file(self, user_id: str, relative_path: str) -> bytes:
        """Read bytes for a stored file after validating ownership and path containment."""

    @abstractmethod
    def delete_file(self, user_id: str, relative_path: str) -> bool:
        """Delete a single user-owned file cleanly."""

    @abstractmethod
    def delete_user_storage(self, user_id: str) -> int:
        """Delete all files owned by a user during account deletion."""

    @abstractmethod
    def list_files(self, user_id: str, bucket: StorageBucket) -> List[str]:
        """List relative paths currently present in a user's bucket."""
