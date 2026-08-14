"""
ADD THIS to llm_schemas.py (paste at the end, alongside the other schemas).
This is the structured-output schema for the judge step: given a GDPR
obligation + retrieved policy evidence, the LLM must return exactly these
fields. article_number/chapter_number/severity/obligation_id are NOT asked
of the LLM -- those come from our own static metadata, only verdict/
confidence/reason/gap are the LLM's judgment call.
"""

from typing import Literal
from pydantic import BaseModel, Field


class JudgeVerdictOutput(BaseModel):
    verdict: Literal["FULLY_MET", "PARTIALLY_MET", "NOT_MET", "CONFLICTING"] = Field(
        description=(
            "FULLY_MET: evidence clearly and completely satisfies the obligation. "
            "PARTIALLY_MET: evidence addresses the obligation but is incomplete/vague. "
            "NOT_MET: no relevant evidence, or evidence clearly does not satisfy the obligation. "
            "CONFLICTING: different evidence chunks contradict each other on this obligation."
        )
    )
    confidence: float = Field(description="0.0 to 1.0 confidence in this verdict, based on evidence relevance/clarity.")
    reason: str = Field(description="1-2 sentence explanation of why this verdict was chosen, referencing the evidence.")
    gap: str | None = Field(
        default=None,
        description="If not FULLY_MET: what specifically is missing or unclear. Null if FULLY_MET."
    )