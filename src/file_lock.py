from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class FileLockError(RuntimeError):
    pass


@dataclass
class FileLock:
    """Cross-platform best-effort exclusive lock.

    Purpose:
      - Gate a single ingestor cycle (including network fetch) without blocking
        read-only consumers (dashboard / replay / validator).
      - Separate from the DB writer lock, which is held only during DB writes.

    This lock is intended to be short-lived (seconds to <1 minute).
    """
    path: str
    _fh: Optional[object] = None

    def acquire(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "a+")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except Exception as e:
            try:
                fh.close()
            finally:
                raise FileLockError(f"Could not acquire lock {self.path}: {e}") from e
        self._fh = fh

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            fh.close()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
