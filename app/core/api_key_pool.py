# app/core/api_key_pool.py

from __future__ import annotations

import itertools
import threading


class APIKeyPool:
    """
    Thread-safe round-robin pool of API keys.

    Why this exists:
        With bounded two-level parallelism (article workers x
        group workers), several LLM calls happen at the same
        moment. A single OpenRouter API key has its own rate
        limit, so all those concurrent calls competing for one
        key's quota is what causes 429 errors under load.

        Spreading calls across multiple keys means each key only
        sees a fraction of the concurrent traffic. This pool is
        the single coordination point that makes that spreading
        thread-safe: every worker thread calls next_key() and
        gets keys handed out round-robin, so no extra
        per-thread bookkeeping is needed anywhere else.

    Usage:
        pool = APIKeyPool(["key_a", "key_b", "key_c"])
        key = pool.next_key()   # call this once per outbound request
    """

    def __init__(self, keys: list[str]) -> None:

        # De-duplicate while preserving order, and drop blanks
        # (e.g. from a trailing comma in an env var).
        cleaned = list(
            dict.fromkeys(
                key.strip()
                for key in keys
                if key and key.strip()
            )
        )

        if not cleaned:
            raise ValueError(
                "APIKeyPool requires at least one non-empty API key."
            )

        self._keys = cleaned
        self._cycle = itertools.cycle(self._keys)
        self._lock = threading.Lock()

    def next_key(self) -> str:
        """Return the next key in round-robin order. Thread-safe."""

        with self._lock:
            return next(self._cycle)

    @property
    def size(self) -> int:
        return len(self._keys)

    @staticmethod
    def mask(key: str) -> str:
        """Last 4 characters only, for safe logging."""

        if len(key) <= 4:
            return "****"

        return f"...{key[-4:]}"