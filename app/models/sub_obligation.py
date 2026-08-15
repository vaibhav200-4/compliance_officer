from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_STATUSES = {
    "MET",
    "PARTIALLY_MET",
    "NOT_MET",
    "CONFLICTING",
    "INSUFFICIENT_EVIDENCE",
    "NOT_APPLICABLE",
}


@dataclass(frozen=True)
class EvidenceReference:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class SubObligationVerdict:
    obligation_id: str
    status: str
    reason: str
    evidence: tuple[EvidenceReference, ...]
    confidence: float

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {self.status}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )