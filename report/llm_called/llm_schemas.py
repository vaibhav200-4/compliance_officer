"""
llm_schemas.py
Pydantic schemas ONLY for LLM structured-output (with_structured_output).
Kept separate from models.py (the report contract) on purpose: these are the
shape of what the LLM must return; models.py is the shape of the final report.
"""

from pydantic import BaseModel, Field


class RequirementEnrichmentItem(BaseModel):
    sub_id: str = Field(description="Must exactly match one of the sub_ids given in the input.")
    analysis: str = Field(description="1-3 sentence interpretation of how the evidence does/doesn't satisfy the requirement, referencing the noted gap. No new facts.")
    fix_required: str = Field(description="One concise, imperative, actionable remediation instruction. Specific, not generic advice.")


class ChapterEnrichmentBatch(BaseModel):
    items: list[RequirementEnrichmentItem]


class ChapterNarrativeOutput(BaseModel):
    narrative: str = Field(description="1-2 sentence callout explaining why this chapter is the most critical gap area, in the tone of a compliance report.")


class ExecutiveSummaryOutput(BaseModel):
    summary: str = Field(description="A single professional paragraph (5-8 sentences) summarizing overall compliance posture, strongest areas, most critical gaps, and urgency of remediation.")

"""
ADD THIS to llm_schemas.py (paste at the end, alongside the other schemas).
This is the structured-output schema for the judge step: given a GDPR
obligation + retrieved policy evidence, the LLM must return exactly these
fields. article_number/chapter_number/severity/obligation_id are NOT asked
of the LLM -- those come from our own static metadata, only verdict/
confidence/reason/gap are the LLM's judgment call.
"""

"""from typing import Literal
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
    )"""

"""
ADD THIS to llm_schemas.py (instead of / in addition to JudgeVerdictOutput
from before). This version returns a LIST of verdicts in one LLM call,
one item per obligation in the batch.
"""

from typing import Literal
from pydantic import BaseModel, Field


class JudgeVerdictBatchItem(BaseModel):
    obligation_id: str = Field(description="Must exactly match one of the obligation_ids given in the input, unchanged.")
    verdict: Literal["FULLY_MET", "PARTIALLY_MET", "NOT_MET", "CONFLICTING"] = Field(
        description=(
            "FULLY_MET: evidence clearly and completely satisfies the obligation. "
            "PARTIALLY_MET: evidence addresses the obligation but is incomplete/vague. "
            "NOT_MET: no relevant evidence, or evidence clearly does not satisfy the obligation. "
            "CONFLICTING: different evidence chunks contradict each other on this obligation."
        )
    )
    confidence: float = Field(description="0.0 to 1.0 confidence in this verdict.")
    reason: str = Field(description="1-2 sentence explanation referencing the evidence.")
    gap: str | None = Field(default=None, description="If not FULLY_MET: what specifically is missing. Null if FULLY_MET.")


class JudgeVerdictBatchOutput(BaseModel):
    items: list[JudgeVerdictBatchItem] = Field(
        description="One item for EVERY obligation given in the input, in any order, matched by obligation_id."
    )