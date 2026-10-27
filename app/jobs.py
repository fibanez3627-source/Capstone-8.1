"""Index lock: one indexing at a time per index folder (lock file next to the manifest,
works on Windows and Linux), so two consoles never run `update` / `rebuild` together.
A `chat` session open in another console picks up the new index via maybe_reload().
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.retriever import HybridEvaluationRetriever

log = logging.getLogger("rag.jobs")
STALE_LOCK_SECS = 6 * 3600


class IndexLock:
    """Cross-process lock via O_EXCL file creation (no fcntl, so it works on Windows)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._held = False

    def acquire(self) -> bool:
        try:
            if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > STALE_LOCK_SECS:
                os.remove(self.path)                    # left over by a crashed process
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
            os.close(fd)
            self._held = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._held:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self._held = False

    def __enter__(self) -> "IndexLock":
        if not self.acquire():
            raise RuntimeError(f"another indexing is running (lock file {self.path})")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def index_lock_for(retriever: HybridEvaluationRetriever) -> IndexLock:
    """The lock lives next to the manifest."""
    return IndexLock(Path(retriever._manifest_file).resolve().parent / "index.lock")
