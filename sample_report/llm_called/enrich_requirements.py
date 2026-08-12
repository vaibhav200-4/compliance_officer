
from langchain_core.prompts import ChatPromptTemplate

from models import Requirement
from llm_schemas import ChapterEnrichmentBatch
from llm_client import get_llm

SYSTEM_PROMPT = """You are a GDPR compliance analyst writing entries for a formal gap-analysis report.
For EACH requirement given, produce:
- analysis: 1-3 sentences interpreting how well the policy evidence satisfies the requirement, referencing the noted gap. Do not invent facts beyond what's given.
- fix_required: ONE concise, imperative, specific remediation instruction (e.g. "Add a 'Right to Data Portability' section documenting..."), not generic advice like "improve compliance."

Return an entry for every sub_id given, in the same order. Do not skip any."""

USER_TEMPLATE = """Chapter: {chapter_name} ({article_range})

Requirements needing analysis:
{requirements_block}
"""


def _format_requirement(req: Requirement) -> str:
    return (
        f"- sub_id: {req.sub_id}\n"
        f"  Article: {req.article_ref}\n"
        f"  GDPR Requires: {req.gdpr_requires}\n"
        f"  Verdict: {req.verdict_label}\n"
        f"  Your Policy Evidence: {req.your_policy}\n"
        f"  Noted Gap: {req.gap or 'N/A'}\n"
    )


def enrich_requirements_for_chapter(
    chapter_name: str,
    article_range: str,
    requirements: list[Requirement],
) -> dict[str, dict]:
    """
    Returns {sub_id: {"analysis": ..., "fix_required": ...}} for the given requirements.
    Only call this with non-FULLY_MET requirements (filter before calling).
    Returns {} immediately if the list is empty (no LLM call made).
    """
    if not requirements:
        return {}

    llm = get_llm().with_structured_output(ChapterEnrichmentBatch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE),
    ])

    requirements_block = "\n".join(_format_requirement(r) for r in requirements)
    chain = prompt | llm
    result: ChapterEnrichmentBatch = chain.invoke({
        "chapter_name": chapter_name,
        "article_range": article_range,
        "requirements_block": requirements_block,
    })

    return {item.sub_id: {"analysis": item.analysis, "fix_required": item.fix_required}
            for item in result.items}