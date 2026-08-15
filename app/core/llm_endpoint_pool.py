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

    def next_endpoint(self) -> LLMEndpoint:
        """Return the next endpoint in round-robin order. Thread-safe."""

        with self._lock:
            endpoint = self._endpoints[self._index]
            self._index = (self._index + 1) % len(self._endpoints)
            return endpoint

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

        Add more providers by following the same pattern -- any
        OpenAI-compatible chat-completions endpoint works.
        """

        endpoints: list[LLMEndpoint] = []

        # --------------------------------------------------------
        # OpenRouter
        # --------------------------------------------------------

        openrouter_model = os.getenv("OPENROUTER_MODEL")

        openrouter_keys = (
            os.getenv("OPENROUTER_API_KEYS")
            or os.getenv("OPENROUTER_API_KEY")
        )

        if openrouter_model and openrouter_keys:

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

        if nim_model and nim_keys:

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
                "NVIDIA_NIM_MODEL + NVIDIA_NIM_API_KEYS."
            )

        return cls(endpoints)
