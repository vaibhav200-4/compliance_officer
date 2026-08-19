"""
Internal exceptions for GDPR compliance analysis.

Classification helps differentiate between retryable transient errors
(network, 429, 5xx, malformed JSON) and non-retryable bugs/configuration issues.
"""

from __future__ import annotations


class ComplianceError(Exception):
    """Base exception for all compliance system errors."""


class ConfigurationError(ComplianceError):
    """Non-retryable configuration error (missing keys, bad settings)."""


class EndpointError(ComplianceError):
    """Base error for LLM endpoint failures."""


class RateLimitError(EndpointError):
    """HTTP 429 Rate Limit error (retryable, triggers endpoint cooldown)."""


class TransientLLMError(EndpointError):
    """HTTP 5xx, connection, or timeout error (retryable)."""


class InvalidLLMResponseError(EndpointError):
    """Malformed JSON, missing fields, or incomplete schema from LLM (retryable)."""


class EvidenceValidationError(ComplianceError):
    """Failed chunk_id or quote validation (retryable)."""


class RetrievalError(ComplianceError):
    """Pinecone or retrieval failure."""
