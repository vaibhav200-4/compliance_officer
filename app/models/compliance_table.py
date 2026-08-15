from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComplianceRow:
    article: str
    requirement_group: str
    principle: str
    status: str
    confidence: float
    evidence: tuple[dict[str, Any], ...]
    reason: str
    gap: str | None
    citation: str | None = None


@dataclass
class ComplianceTable:
    rows: list[ComplianceRow]

    def add(self, row: ComplianceRow) -> None:
        self.rows.append(row)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "article": row.article,
                "requirement_group": row.requirement_group,
                "principle": row.principle,
                "status": row.status,
                "confidence": row.confidence,
                "evidence": list(row.evidence),
                "reason": row.reason,
                "gap": row.gap,
                "citation": row.citation,
            }
            for row in self.rows
        ]