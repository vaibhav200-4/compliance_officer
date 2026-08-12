"""
enrich_executive_summary.py
Runs LAST, after all requirement + chapter enrichment is done, because it needs
the full merged picture (overall score, chapter breakdown, and the P1 items)
to write a coherent summary — not the raw aggregator JSON.
"""

from langchain_core.prompts import ChatPromptTemplate

from models import ReportData
from llm_schemas import ExecutiveSummaryOutput
from llm_client import get_llm

SYSTEM_PROMPT = """You are a GDPR compliance analyst writing the Executive Summary section
of a formal gap-analysis report for company leadership. Write ONE professional paragraph
(5-8 sentences) covering: overall compliance posture, which areas are strong, which chapters/
rights represent the most critical gaps and why they matter legally, and the urgency of
remediation. Ground every claim in the data given — do not invent specifics not present below."""

USER_TEMPLATE = """Company: {company}
Overall Score: {overall_score}%
Totals: {fully_met} fully met, {partially_met} partially met, {not_met} not met, {conflicting} conflicting (of {total})
Risk Level: {risk_level}

Chapter scores:
{chapter_lines}

Top critical (P1) gaps:
{p1_lines}
"""


def generate_executive_summary(data: ReportData) -> str:
    llm = get_llm().with_structured_output(ExecutiveSummaryOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE),
    ])

    chapter_lines = "\n".join(
        f"- Ch {c.chapter} {c.name}: {c.score if c.score is not None else 'Not Assessed'}%"
        for c in data.chapters
    )
    p1_items = [a for a in data.priority_actions if a.priority == "P1"]
    p1_lines = "\n".join(
        f"- {a.sub_id} ({a.article_ref}): {a.action_required}" for a in p1_items
    ) or "None"

    chain = prompt | llm
    result: ExecutiveSummaryOutput = chain.invoke({
        "company": data.meta.company,
        "overall_score": data.overall.score,
        "fully_met": data.overall.fully_met,
        "partially_met": data.overall.partially_met,
        "not_met": data.overall.not_met,
        "conflicting": data.overall.conflicting,
        "total": data.overall.total,
        "risk_level": data.overall.risk_level,
        "chapter_lines": chapter_lines,
        "p1_lines": p1_lines,
    })
    return result.summary