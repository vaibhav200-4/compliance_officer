"""
models.py
Pydantic data model = single contract for the report.
Track A (this batch) fills everything EXCEPT the fields marked "LLM (Track B)".
Those fields default to None / "" so the PDF can still render before Track B runs.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    page: Optional[int] = None
    similarity: Optional[float] = None


class Requirement(BaseModel):
    sub_id: str
    chapter: int
    article_ref: str                       # Track A (static lookup)
    verdict: str                           # FULLY_MET / PARTIALLY_MET / NOT_MET / CONFLICTING
    verdict_label: str                     # "Fully Met" etc, human readable
    verdict_icon: str                      # "check" / "partial" / "cross" / "conflict"
    confidence: Optional[float] = None
    confidence_label: str = "Low"          # Track A (thresholded)
    evidence: list[Evidence] = Field(default_factory=list)
    gdpr_requires: str = ""                # Track A (static legal-text lookup)
    your_policy: str = ""                  # Track A (= evidence[0].text, verbatim)
    gap: Optional[str] = None              # from aggregator (kept as-is, informational)

    # ---- LLM (Track B) ----
    analysis: str = ""                     # LLM
    fix_required: str = ""                 # LLM


class ChapterScore(BaseModel):
    chapter: int
    name: str                              # Track A (static lookup)
    article_range: str                     # Track A (static lookup)
    score: Optional[float] = None
    status: str = "Not Assessed"
    total: int = 0
    fully_met: int = 0
    partially_met: int = 0
    not_met: int = 0
    conflicting: int = 0

    # ---- LLM (Track B) ----
    narrative: str = ""                    # 1-2 line callout, only for weak chapters


class OverallSummary(BaseModel):
    score: Optional[float] = None
    total: int = 0
    fully_met: int = 0
    partially_met: int = 0
    not_met: int = 0
    conflicting: int = 0
    risk_level: str = "Not Assessed"       # Track A (threshold rule)
    max_fine_exposure: str = "€20M or 4% annual revenue"  # static, GDPR fixed text


class Meta(BaseModel):
    company: str
    policy_analyzed: str
    standard: str = "GDPR — All 99 Articles, 11 Chapters"
    analysis_date: str
    generated_by: str = "ComplianceIQ Engine v1.0 — onetab.ai"
    analyzed_by: str = "Agentic RAG Pipeline (4-Agent System)"


class PriorityAction(BaseModel):
    priority: str                          # "P1" / "P2" / "P3"
    sub_id: str
    chapter: int
    article_ref: str
    action_required: str                   # Track A placeholder = gap text
                                            # (Track B later overwrites with fix_required first line)
    current_status: str


class ReportData(BaseModel):
    meta: Meta
    overall: OverallSummary
    chapters: list[ChapterScore] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    priority_actions: list[PriorityAction] = Field(default_factory=list)

    # ---- LLM (Track B) ----
    executive_summary: str = ""

    # ---- Track A computed extras for template convenience ----
    score_ring_offset: float = 0.0         # precomputed SVG stroke-dashoffset
    score_ring_circumference: float = 0.0