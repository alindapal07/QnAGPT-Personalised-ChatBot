"""Local filesystem storage provider enforcing per-user containment and checksum verification."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import List
import uuid

from qna_bot.core.exceptions import SecurityError, StorageError
from qna_bot.storage.base import STORAGE_BUCKETS, StorageBucket, StorageProvider, StoredFileMeta

_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_user_segment(user_id: str) -> str:
    raw = (user_id or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw:
        raise SecurityError("Invalid user identifier for storage isolation.")
    cleaned = _SAFE_SEGMENT_RE.sub("_", raw).strip("._")
    if not cleaned or cleaned in (".", ".."):
        raise SecurityError("Invalid user identifier for storage isolation.")
    return cleaned


def _sanitize_filename(filename: str) -> str:
    # Reject explicit traversal markers early before taking basename
    raw = str(filename or "").replace("\\", "/")
    if ".." in raw.split("/"):
        raise SecurityError("Filename contains disallowed directory traversal segments.")
    base = Path(raw).name.strip()
    if not base or base in (".", ".."):
        raise StorageError("Uploaded filename is empty or invalid.")
    safe = _SAFE_SEGMENT_RE.sub("_", base).strip("_")
    if not safe:
        safe = "file.bin"
    return safe[:160]


class LocalStorageProvider(StorageProvider):
    """Stores user documents, images, attachments, and exports under storage/users/{user_id}/."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.users_root = (self.root_dir / "users").resolve()
        self.backups_root = (self.root_dir / "backups").resolve()
        self.users_root.mkdir(parents=True, exist_ok=True)
        self.backups_root.mkdir(parents=True, exist_ok=True)

    def _user_root(self, user_id: str) -> Path:
        safe_user = _sanitize_user_segment(user_id)
        target = (self.users_root / safe_user).resolve()
        if not target.is_relative_to(self.users_root) or target == self.users_root:
            raise SecurityError("Resolved user storage directory escapes users root.")
        return target

    def resolve_user_path(self, user_id: str, relative_path: str) -> Path:
        """Resolve a stored relative path strictly inside the authenticated user's directory."""
        user_root = self._user_root(user_id)
        raw_rel = str(relative_path or "").strip().replace("\\", "/")
        if not raw_rel or raw_rel.startswith("/") or ":" in raw_rel.split("/")[0]:
            raise SecurityError("Storage path must be a relative path within the user workspace.")
        candidate = (user_root / raw_rel).resolve()
        if not candidate.is_relative_to(user_root):
            raise SecurityError("Storage path escapes authenticated user directory.")
        return candidate

    def save_file(
        self,
        user_id: str,
        bucket: StorageBucket,
        filename: str,
        data: bytes,
    ) -> StoredFileMeta:
        if bucket not in STORAGE_BUCKETS:
            raise StorageError(f"Unsupported storage bucket '{bucket}'.")
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise StorageError("Cannot store empty file payload.")

        user_root = self._user_root(user_id)
        bucket_dir = (user_root / bucket).resolve()
        if not bucket_dir.is_relative_to(user_root):
            raise SecurityError("Storage bucket escapes authenticated user directory.")
        bucket_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_filename(filename)
        unique_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
        dest_path = (bucket_dir / unique_name).resolve()
        if not dest_path.is_relative_to(bucket_dir):
            raise SecurityError("Destination file escapes storage bucket.")

        payload = bytes(data)
        checksum = hashlib.sha256(payload).hexdigest()

        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".upload_", dir=str(bucket_dir))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                os.replace(tmp_name, dest_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise StorageError(f"Failed to persist file '{safe_name}': {exc}") from exc

        rel_path = dest_path.relative_to(user_root).as_posix()
        return StoredFileMeta(
            relative_path=rel_path,
            absolute_path=str(dest_path),
            filename=safe_name,
            bucket=bucket,
            size_bytes=len(payload),
            checksum=checksum,
        )

    def read_file(self, user_id: str, relative_path: str) -> bytes:
        path = self.resolve_user_path(user_id, relative_path)
        if not path.is_file():
            raise StorageError(f"Stored file '{relative_path}' was not found.")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StorageError(f"Failed to read stored file '{relative_path}': {exc}") from exc

    def delete_file(self, user_id: str, relative_path: str) -> bool:
        path = self.resolve_user_path(user_id, relative_path)
        if not path.exists():
            return False
        if not path.is_file():
            raise SecurityError("Refusing to delete non-file storage target.")
        try:
            path.unlink()
            return True
        except OSError as exc:
            raise StorageError(f"Failed to delete stored file '{relative_path}': {exc}") from exc

    def delete_user_storage(self, user_id: str) -> int:
        user_root = self._user_root(user_id)
        if not user_root.exists():
            return 0
        count = sum(1 for item in user_root.rglob("*") if item.is_file())
        try:
            shutil.rmtree(user_root)
        except OSError as exc:
            raise StorageError(f"Failed to clean user storage directory: {exc}") from exc
        return count

    def list_files(self, user_id: str, bucket: StorageBucket) -> List[str]:
        if bucket not in STORAGE_BUCKETS:
            raise StorageError(f"Unsupported storage bucket '{bucket}'.")
        user_root = self._user_root(user_id)
        bucket_dir = (user_root / bucket).resolve()
        if not bucket_dir.is_dir():
            return []
        return sorted(
            item.relative_to(user_root).as_posix()
            for item in bucket_dir.iterdir()
            if item.is_file()
        )
