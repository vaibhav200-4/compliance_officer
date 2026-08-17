from datetime import date

from report.templated.models import (
    ReportData, Meta, OverallSummary, ChapterScore, Requirement, Evidence,
)
from report.templated.static_data import (
    CHAPTER_META, VERDICT_META, get_requirement_meta,
    confidence_label, chapter_status, overall_risk_level,
)
from report.templated.priority_engine import (
    build_priority_actions, refine_priority_actions_with_fixes,
)
from report.llm_called.enrich_requirements import enrich_requirements_for_chapter


def _score_from_counts(fully: int, partial: int, total: int) -> float | None:
    """Matches the aggregator's own formula: (fully + 0.5*partial) / total * 100."""
    if total == 0:
        return None
    return round((fully + 0.5 * partial) / total * 100, 2)


def build_report_data(
    aggregator_output: list[dict],
    meta_overrides: dict | None = None,
    enrich: bool = True,
    max_requirements: int | None = None,
    max_priority_per_tier: int | None = 5,
) -> ReportData:
    """
    enrich: when True (default), calls Track B (enrich_requirements_for_chapter)
        for every non-FULLY_MET requirement, per chapter, and uses the result to
        fill in `analysis` / `fix_required`, then upgrades the priority-action
        wording via refine_priority_actions_with_fixes. Set False to skip LLM
        calls entirely (e.g. for a quick Track-A-only draft).
    max_requirements: if set, keep only this many requirements in the final
        report -- the most severe/highest-confidence ones -- instead of all of
        them. Chapters/overall totals are unaffected (still computed from the
        FULL aggregator_output), only the printed requirement cards are trimmed.
    max_priority_per_tier: cap on how many P1/P2/P3 rows are kept (None = no cap).
    """
    meta_overrides = meta_overrides or {}
    meta = Meta(
        company=meta_overrides.get("company", "TechStartup Pvt Ltd"),
        policy_analyzed=meta_overrides.get("policy_analyzed", "Privacy Policy v2.1"),
        analysis_date=meta_overrides.get("analysis_date", date.today().strftime("%d %B %Y")),
    )

    chapters: list[ChapterScore] = []
    requirements: list[Requirement] = []

    tot_total = tot_full = tot_partial = tot_notmet = tot_conflict = tot_notapplicable = 0

    for block in aggregator_output:
        chap_info = block["chapter_scores"][0]
        chap_num = chap_info["chapter"]
        summary = block["summary"]
        cmeta = CHAPTER_META.get(chap_num, {"name": f"Chapter {chap_num}", "articles": "-"})
        chap_name = cmeta["name"]
        article_range = cmeta["articles"]

        chapters.append(ChapterScore(
            chapter=chap_num,
            name=chap_name,
            article_range=article_range,
            score=block["overall_score"],
            status=chapter_status(block["overall_score"]),
            total=summary["total"],
            fully_met=summary["fully_met"],
            partially_met=summary["partially_met"],
            not_met=summary["not_met"],
            conflicting=summary["conflicting"],
            not_applicable=summary.get("not_applicable", 0),
        ))

        tot_total += summary["total"]
        tot_full += summary["fully_met"]
        tot_partial += summary["partially_met"]
        tot_notmet += summary["not_met"]
        tot_conflict += summary["conflicting"]
        tot_notapplicable += summary.get("not_applicable", 0)

        chapter_requirements: list[Requirement] = []
        for req in block["requirements"]:
            vmeta = VERDICT_META.get(req["verdict"], {"label": req["verdict"], "icon": "cross", "css": ""})
            rmeta = get_requirement_meta(req["sub_id"])
            evidence_list = [Evidence(**e) for e in req.get("evidence", [])]
            your_policy = evidence_list[0].text if evidence_list else "No matching evidence found in policy."

            chapter_requirements.append(Requirement(
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

        # ---- Track B: enrich this chapter's non-FULLY_MET requirements ----
            # ---- Track B: enrich this chapter's requirements that actually need analysis ----
            # (skip FULLY_MET -- already compliant; skip NOT_APPLICABLE -- no gap to analyze)
        if enrich:
            needs_enrichment = [
                r for r in chapter_requirements
                if r.verdict not in ("FULLY_MET", "NOT_APPLICABLE")
            ]
            if needs_enrichment:
                print(f"  Enriching {len(needs_enrichment)} requirement(s) in {chap_name}...")
                enriched = enrich_requirements_for_chapter(chap_name, article_range, needs_enrichment)
                for r in chapter_requirements:
                    hit = enriched.get(r.sub_id)
                    if hit:
                        r.analysis = hit["analysis"]
                        r.fix_required = hit["fix_required"]

        requirements.extend(chapter_requirements)

    overall_score = _score_from_counts(tot_full, tot_partial, tot_total)
    overall = OverallSummary(
        score=overall_score,
        total=tot_total,
        fully_met=tot_full,
        partially_met=tot_partial,
        not_met=tot_notmet,
        conflicting=tot_conflict,
        not_applicable=tot_notapplicable,
        risk_level=overall_risk_level(overall_score),
    )

    priority_actions = build_priority_actions(requirements)
    # ---- upgrade placeholder priority-action text with the enriched fix_required ----
    if enrich:
        priority_actions = refine_priority_actions_with_fixes(priority_actions, requirements)

    if max_priority_per_tier is not None:
        capped: list = []
        seen_per_tier: dict[str, int] = {"P1": 0, "P2": 0, "P3": 0}
        for a in priority_actions:
            if seen_per_tier.get(a.priority, 0) >= max_priority_per_tier:
                continue
            seen_per_tier[a.priority] = seen_per_tier.get(a.priority, 0) + 1
            capped.append(a)
        priority_actions = capped

    report_requirements = requirements
    if max_requirements is not None and len(requirements) > max_requirements:
        # Balanced trim: keep a proportional mix across verdict types instead
        # of just the top-N most severe, so the report shows curated examples
        # from each category (Not Met, Partially Met, Fully Met, Conflicting)
        # rather than only worst-case items. Higher confidence kept first
        # within each bucket.
        from collections import defaultdict

        buckets: dict[str, list] = defaultdict(list)
        for r in requirements:
            buckets[r.verdict].append(r)
        for v in buckets:
            buckets[v].sort(key=lambda r: -(r.confidence or 0))

        # Proportional split of max_requirements across verdict types.
        # Adjust these ratios if you want a different mix.
        ratios = {"NOT_MET": 0.5, "PARTIALLY_MET": 0.35, "FULLY_MET": 0.10, "CONFLICTING": 0.05}
        caps = {v: max(1, round(max_requirements * ratio)) for v, ratio in ratios.items() if buckets.get(v)}

        # Trim caps down if they overshoot max_requirements due to rounding.
        while sum(caps.values()) > max_requirements:
            largest = max(caps, key=lambda v: caps[v])
            caps[largest] -= 1

        report_requirements = []
        for v, cap in caps.items():
            report_requirements.extend(buckets.get(v, [])[:cap])

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
        requirements=report_requirements,
        priority_actions=priority_actions,
        score_ring_circumference=circumference,
        score_ring_offset=ring_offset,
    )