"""
build_report_data.py
Converts the raw list-of-chapter-dicts from aggregate_func.py into a validated
ReportData object. This is 100% deterministic Python — no LLM calls anywhere
in this file. Track B (later) will take the returned ReportData and only fill
in: requirement.analysis, requirement.fix_required, chapter.narrative,
report.executive_summary, and refine priority_actions[].action_required.
"""

from datetime import date

from models import (
    ReportData, Meta, OverallSummary, ChapterScore, Requirement, Evidence,
)
from static_data import (
    CHAPTER_META, VERDICT_META, get_requirement_meta,
    confidence_label, chapter_status, overall_risk_level,
)
from priority_engine import build_priority_actions


def _score_from_counts(fully: int, partial: int, total: int) -> float | None:
    """Matches the aggregator's own formula: (fully + 0.5*partial) / total * 100."""
    if total == 0:
        return None
    return round((fully + 0.5 * partial) / total * 100, 2)


def build_report_data(aggregator_output: list[dict], meta_overrides: dict | None = None) -> ReportData:
    meta_overrides = meta_overrides or {}
    meta = Meta(
        company=meta_overrides.get("company", "TechStartup Pvt Ltd"),
        policy_analyzed=meta_overrides.get("policy_analyzed", "Privacy Policy v2.1"),
        analysis_date=meta_overrides.get("analysis_date", date.today().strftime("%d %B %Y")),
    )

    chapters: list[ChapterScore] = []
    requirements: list[Requirement] = []

    tot_total = tot_full = tot_partial = tot_notmet = tot_conflict = 0

    for block in aggregator_output:
        chap_info = block["chapter_scores"][0]
        chap_num = chap_info["chapter"]
        summary = block["summary"]
        cmeta = CHAPTER_META.get(chap_num, {"name": f"Chapter {chap_num}", "articles": "-"})

        chapters.append(ChapterScore(
            chapter=chap_num,
            name=cmeta["name"],
            article_range=cmeta["articles"],
            score=block["overall_score"],
            status=chapter_status(block["overall_score"]),
            total=summary["total"],
            fully_met=summary["fully_met"],
            partially_met=summary["partially_met"],
            not_met=summary["not_met"],
            conflicting=summary["conflicting"],
        ))

        tot_total += summary["total"]
        tot_full += summary["fully_met"]
        tot_partial += summary["partially_met"]
        tot_notmet += summary["not_met"]
        tot_conflict += summary["conflicting"]

        for req in block["requirements"]:
            vmeta = VERDICT_META.get(req["verdict"], {"label": req["verdict"], "icon": "cross", "css": ""})
            rmeta = get_requirement_meta(req["sub_id"])
            evidence_list = [Evidence(**e) for e in req.get("evidence", [])]
            your_policy = evidence_list[0].text if evidence_list else "No matching evidence found in policy."

            requirements.append(Requirement(
                sub_id=req["sub_id"],
                chapter=chap_num,
                article_ref=rmeta["article_ref"],
                verdict=req["verdict"],
                verdict_label=vmeta["label"],
                verdict_icon=vmeta["icon"],
                confidence=req.get("confidence"),
                confidence_label=confidence_label(req.get("confidence")),
                evidence=evidence_list,
                gdpr_requires=rmeta["requires"],
                your_policy=your_policy,
                gap=req.get("gap"),
            ))

    overall_score = _score_from_counts(tot_full, tot_partial, tot_total)
    overall = OverallSummary(
        score=overall_score,
        total=tot_total,
        fully_met=tot_full,
        partially_met=tot_partial,
        not_met=tot_notmet,
        conflicting=tot_conflict,
        risk_level=overall_risk_level(overall_score),
    )

    priority_actions = build_priority_actions(requirements)

    # SVG ring geometry (r=54 -> circumference ~339.3), precomputed so the
    # Jinja template just drops in numbers, no math in the template.
    radius = 54
    circumference = round(2 * 3.14159265 * radius, 2)
    pct = (overall_score or 0) / 100
    ring_offset = round(circumference * (1 - pct), 2)

    return ReportData(
        meta=meta,
        overall=overall,
        chapters=chapters,
        requirements=requirements,
        priority_actions=priority_actions,
        score_ring_circumference=circumference,
        score_ring_offset=ring_offset,
    )