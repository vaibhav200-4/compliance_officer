"""
enrich_chapters.py
Generates the 1-2 line italic callout under the chapter table (like the
"Chapter III (Rights of the Data Subject) is the most critical gap area..."
line in the original report). Only called for chapters that need attention —
skips chapters that are None/Not Assessed or already scoring well.
"""

from langchain_core.prompts import ChatPromptTemplate

from models import ChapterScore
from llm_schemas import ChapterNarrativeOutput
from llm_client import get_llm

NARRATIVE_THRESHOLD = 70  # only narrate chapters scoring below this

SYSTEM_PROMPT = """You are a GDPR compliance analyst. Write ONE short callout (1-2 sentences)
explaining why this chapter is a critical gap area in a formal compliance report. Reference
what the chapter covers and why it matters (e.g. what regulators scrutinize). Do not repeat
the score number, it's already shown elsewhere."""

USER_TEMPLATE = """Chapter {chapter_num}: {name} ({article_range})
Score: {score}%
Breakdown: {fully_met} fully met, {partially_met} partially met, {not_met} not met, {conflicting} conflicting (of {total} total)
"""


def needs_narrative(chapter: ChapterScore) -> bool:
    return chapter.score is not None and chapter.score < NARRATIVE_THRESHOLD


def enrich_chapter_narrative(chapter: ChapterScore) -> str:
    """Returns the narrative string. Caller should check needs_narrative() first
    to avoid wasting a call on chapters that don't need one."""
    llm = get_llm().with_structured_output(ChapterNarrativeOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE),
    ])
    chain = prompt | llm
    result: ChapterNarrativeOutput = chain.invoke({
        "chapter_num": chapter.chapter,
        "name": chapter.name,
        "article_range": chapter.article_range,
        "score": chapter.score,
        "fully_met": chapter.fully_met,
        "partially_met": chapter.partially_met,
        "not_met": chapter.not_met,
        "conflicting": chapter.conflicting,
        "total": chapter.total,
    })
    return result.narrative