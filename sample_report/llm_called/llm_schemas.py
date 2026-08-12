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