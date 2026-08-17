"""
priority_engine.py
Deterministic rule engine: verdict severity x chapter weight -> P1/P2/P3.
Runs on Track A data only (verdict + gap). Track B later can overwrite
`action_required` with the LLM-generated fix_required text, but the
priority BUCKET itself stays rule-based (keeps risk ranking trustworthy/audit-safe).
"""

from report.templated.models import Requirement, PriorityAction

# Chapters covering core, most-scrutinized obligations get extra weight.
# (II Principles, III Data Subject Rights, IV Controller/Processor duties)
HIGH_WEIGHT_CHAPTERS = {2, 3, 4}

SEVERITY = {
    "NOT_MET": 3,
    "CONFLICTING": 3,
    "PARTIALLY_MET": 1,
    "FULLY_MET": 0,
}


def _priority_bucket(verdict: str, chapter: int) -> str | None:
    sev = SEVERITY.get(verdict, 0)
    if sev == 0:
        return None  # FULLY_MET -> no action needed
    if sev == 3 and chapter in HIGH_WEIGHT_CHAPTERS:
        return "P1"
    if sev == 3:
        return "P2"
    # sev == 1 (PARTIALLY_MET)
    if chapter in HIGH_WEIGHT_CHAPTERS:
        return "P2"
    return "P3"


def build_priority_actions(requirements: list[Requirement]) -> list[PriorityAction]:
    actions: list[PriorityAction] = []
    for req in requirements:
        bucket = _priority_bucket(req.verdict, req.chapter)
        if bucket is None:
            continue
        # Placeholder action text from `gap` until Track B fills fix_required.
        action_text = req.gap or f"Address {req.verdict_label.lower()} requirement {req.sub_id}."
        actions.append(PriorityAction(
            priority=bucket,
            sub_id=req.sub_id,
            chapter=req.chapter,
            article_ref=req.article_ref,
            action_required=action_text,
            current_status=req.verdict_label,
        ))

    # Sort P1 -> P2 -> P3, then by chapter for readability
    order = {"P1": 0, "P2": 1, "P3": 2}
    actions.sort(key=lambda a: (order[a.priority], a.chapter, a.sub_id))
    return actions


def refine_priority_actions_with_fixes(
    priority_actions: list[PriorityAction],
    requirements: list[Requirement],
) -> list[PriorityAction]:
    """
    Track B step: once requirement.fix_required has been LLM-filled, swap the
    Track-A placeholder (gap text) for the sharper fix_required text. The
    PRIORITY BUCKET itself is never touched here — only the action_required
    wording is upgraded, keeping the risk ranking rule-based/audit-safe.
    """
    fix_by_sub_id = {r.sub_id: r.fix_required for r in requirements if r.fix_required}
    for action in priority_actions:
        if action.sub_id in fix_by_sub_id:
            action.action_required = fix_by_sub_id[action.sub_id]
    return priority_actions