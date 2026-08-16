# app/core/llm_endpoint_pool.py

from __future__ import annotations

import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMEndpoint:
    """
    One callable LLM endpoint: a specific (provider, key, model).

    Different providers need different base_url/model, so a plain
    key string isn't enough once you mix providers -- this bundles
    everything one outbound request needs.
    """

    provider: str      # e.g. "openrouter", "nvidia"
    base_url: str       # full chat-completions URL
    api_key: str
    model: str


class LLMEndpointPool:
    """
    Thread-safe round-robin pool of LLM endpoints, possibly
    spanning multiple providers.

    Why this exists:
        Each provider key has its own rate limit (OpenRouter free
        tier, NVIDIA NIM's ~40 RPM). Concurrent group/article
        workers all calling next_endpoint() spread load across
        every configured key from every configured provider, so
        no single key's limit becomes the bottleneck.

    Usage:
        pool = LLMEndpointPool([endpoint_a, endpoint_b, ...])
        endpoint = pool.next_endpoint()
    """

    def __init__(self, endpoints: list[LLMEndpoint]) -> None:

        if not endpoints:
            raise ValueError(
                "LLMEndpointPool requires at least one endpoint."
            )

        # A duplicate endpoint does not add useful capacity and would skew
        # round-robin selection.  Keys are only meaningful within a provider,
        # so preserve the first occurrence of each provider/key combination.
        self._endpoints = []
        seen: set[tuple[str, str]] = set()
        for endpoint in endpoints:
            key = endpoint.api_key.strip()
            identity = (endpoint.provider, key)
            if not key or identity in seen:
                continue
            seen.add(identity)
            self._endpoints.append(
                LLMEndpoint(
                    provider=endpoint.provider,
                    base_url=endpoint.base_url,
                    api_key=key,
                    model=endpoint.model,
                )
            )

        if not self._endpoints:
            raise ValueError(
                "LLMEndpointPool requires at least one non-empty endpoint."
            )

        self._index = 0
        self._lock = threading.Lock()
        self._cooldown_until: dict[tuple[str, str], float] = {}
        self._consecutive_failures: dict[tuple[str, str], int] = {}

    def mark_cooldown(
        self,
        endpoint: LLMEndpoint,
        duration_seconds: float = 15.0,
    ) -> None:
        """
        Temporarily put an endpoint into cooldown (e.g. after a 429 rate limit or repeated 5xx).
        Thread-safe.
        """
        import time

        identity = (endpoint.provider, endpoint.api_key.strip())
        with self._lock:
            self._cooldown_until[identity] = time.time() + duration_seconds

    def record_failure(
        self,
        endpoint: LLMEndpoint,
        is_5xx: bool = False,
    ) -> None:
        """
        Record a non-429 failure on an endpoint.
        If consecutive 5xx errors reach threshold, trigger temporary cooldown.
        """
        import time

        identity = (endpoint.provider, endpoint.api_key.strip())
        with self._lock:
            if is_5xx:
                count = self._consecutive_failures.get(identity, 0) + 1
                self._consecutive_failures[identity] = count
                if count >= 3:
                    self._cooldown_until[identity] = time.time() + 15.0
                    self._consecutive_failures[identity] = 0
            else:
                self._consecutive_failures[identity] = 0

    def record_success(self, endpoint: LLMEndpoint) -> None:
        """Clear consecutive failure counter on success."""
        identity = (endpoint.provider, endpoint.api_key.strip())
        with self._lock:
            self._consecutive_failures[identity] = 0

    def next_endpoint(self) -> LLMEndpoint:
        """
        Return the next available endpoint in round-robin order, skipping endpoints in cooldown.
        If all endpoints are in cooldown, returns the one that expires earliest.
        Thread-safe.
        """
        import time

        now = time.time()

        with self._lock:
            n = len(self._endpoints)
            # 1. Try to find the next endpoint not in cooldown
            for offset in range(n):
                idx = (self._index + offset) % n
                ep = self._endpoints[idx]
                identity = (ep.provider, ep.api_key.strip())
                cooldown_time = self._cooldown_until.get(identity, 0.0)
                if now >= cooldown_time:
                    self._index = (idx + 1) % n
                    return ep

            # 2. If all endpoints are in cooldown, pick the one with earliest expiration
            best_ep = min(
                self._endpoints,
                key=lambda ep: self._cooldown_until.get((ep.provider, ep.api_key.strip()), 0.0),
            )
            # Advance index past best_ep
            best_idx = self._endpoints.index(best_ep)
            self._index = (best_idx + 1) % n
            return best_ep

    @property
    def size(self) -> int:
        return len(self._endpoints)

    def providers(self) -> set[str]:
        return {endpoint.provider for endpoint in self._endpoints}

    @staticmethod
    def mask(key: str) -> str:
        """Last 4 characters only, for safe logging."""

        if len(key) <= 4:
            return "****"

        return f"...{key[-4:]}"

    # ============================================================
    # ENV-BASED CONSTRUCTION
    # ============================================================

    @classmethod
    def from_env(cls) -> "LLMEndpointPool":
        """
        Build a pool from environment variables, combining every
        provider that has both a model and at least one key set.

        Supported:

            OPENROUTER_MODEL
            OPENROUTER_API_KEYS   (comma-separated, preferred)
            OPENROUTER_API_KEY    (single key, fallback)

            NVIDIA_NIM_MODEL
            NVIDIA_NIM_API_KEYS   (comma-separated)
            NVIDIA_NIM_BASE_URL   (optional, defaults to NVIDIA's
                                    hosted endpoint)

            LLM_PROVIDER  (optional, filters to a single provider:
                           "nvidia" or "openrouter". If not set,
                           all configured providers are included.)

        Add more providers by following the same pattern -- any
        OpenAI-compatible chat-completions endpoint works.
        """

        endpoints: list[LLMEndpoint] = []

        # Get optional provider filter
        llm_provider_filter = os.getenv("LLM_PROVIDER", "").strip().lower()

        # --------------------------------------------------------
        # OpenRouter
        # --------------------------------------------------------

        openrouter_model = os.getenv("OPENROUTER_MODEL")

        openrouter_keys = (
            os.getenv("OPENROUTER_API_KEYS")
            or os.getenv("OPENROUTER_API_KEY")
        )

        # Include OpenRouter if configured and not filtered out
        if openrouter_model and openrouter_keys:
            if not llm_provider_filter or llm_provider_filter == "openrouter":

                for raw_key in openrouter_keys.split(","):

                    key = raw_key.strip()

                    if not key:
                        continue

                    endpoints.append(
                        LLMEndpoint(
                            provider="openrouter",
                            base_url=(
                                "https://openrouter.ai/api/v1"
                                "/chat/completions"
                            ),
                            api_key=key,
                            model=openrouter_model,
                        )
                    )

        # --------------------------------------------------------
        # NVIDIA NIM
        # --------------------------------------------------------

        nim_model = os.getenv("NVIDIA_NIM_MODEL")
        nim_keys = os.getenv("NVIDIA_NIM_API_KEYS")

        nim_base_url = os.getenv(
            "NVIDIA_NIM_BASE_URL",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )

        # Include NVIDIA if configured and not filtered out
        if nim_model and nim_keys:
            if not llm_provider_filter or llm_provider_filter == "nvidia":

                for raw_key in nim_keys.split(","):

                    key = raw_key.strip()

                    if not key:
                        continue

                    endpoints.append(
                        LLMEndpoint(
                            provider="nvidia",
                            base_url=nim_base_url,
                            api_key=key,
                            model=nim_model,
                        )
                    )

        if not endpoints:
            raise ValueError(
                "No LLM endpoints configured. Set at least one "
                "of: OPENROUTER_MODEL + OPENROUTER_API_KEYS, or "
                "NVIDIA_NIM_MODEL + NVIDIA_NIM_API_KEYS. "
                "If LLM_PROVIDER is set, ensure the corresponding "
                "model and keys are configured."
            )

        return cls(endpoints)
