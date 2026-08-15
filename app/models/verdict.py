from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.sub_obligation import (
    SubObligationVerdict,
)


@dataclass(frozen=True)
class GroupVerdict:
    group_id: str
    principle: str
    status: str
    confidence: float
    sub_obligations: tuple[SubObligationVerdict, ...]
    reason: str
    gap: str | None


def aggregate_status(
    statuses: Iterable[str],
    condition_logic: str,
) -> str:

    statuses = list(statuses)

    if not statuses:
        return "INSUFFICIENT_EVIDENCE"

    if all(
        status == "NOT_APPLICABLE"
        for status in statuses
    ):
        return "NOT_APPLICABLE"

    applicable = [
        status
        for status in statuses
        if status != "NOT_APPLICABLE"
    ]

    if not applicable:
        return "NOT_APPLICABLE"

    if "CONFLICTING" in applicable:
        return "CONFLICTING"

    if condition_logic == "ALL":

        if all(
            status == "MET"
            for status in applicable
        ):
            return "MET"

        if all(
            status == "INSUFFICIENT_EVIDENCE"
            for status in applicable
        ):
            return "INSUFFICIENT_EVIDENCE"

        if any(
            status == "NOT_MET"
            for status in applicable
        ):
            if all(
                status in {
                    "NOT_MET",
                    "PARTIALLY_MET",
                }
                for status in applicable
            ):
                return "PARTIALLY_MET"

        return "PARTIALLY_MET"

    if condition_logic == "ANY":

        if any(
            status == "MET"
            for status in applicable
        ):
            return "MET"

        if all(
            status == "INSUFFICIENT_EVIDENCE"
            for status in applicable
        ):
            return "INSUFFICIENT_EVIDENCE"

        if all(
            status == "NOT_MET"
            for status in applicable
        ):
            return "NOT_MET"

        return "PARTIALLY_MET"

    if condition_logic == "SINGLE":

        return applicable[0]

    raise ValueError(
        f"Unsupported condition logic: {condition_logic}"
    )