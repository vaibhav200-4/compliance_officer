"""
run_track2.py
Takes the Track A ReportData (already validated, already has score/counts/tables)
and layers LLM enrichment on top: per-chapter requirement analysis+fix, chapter
narratives for weak chapters, and the executive summary (run last). Then
re-renders the final PDF.

Run this AFTER run_track1.py has produced report_data once — or just run this
directly, it rebuilds Track A first then enriches.
"""

import json

from sample_report.templated.build_report_data import build_report_data
from sample_report.llm_called.enrich_requirements import enrich_requirements_for_chapter
from sample_report.llm_called.enrich_chapters import needs_narrative, enrich_chapter_narrative
from sample_report.llm_called.enrich_executive_summary import generate_executive_summary
from sample_report.templated.priority_engine import refine_priority_actions_with_fixes
from sample_report.templated.render_pdf import render_report_pdf   # swap to render_pdf_playwright if needed


def enrich_report(report_data):
    # ---- Step 1: per-chapter requirement analysis + fix_required ----
    for chapter in report_data.chapters:
        if chapter.total == 0:
            continue  # "Not Assessed" chapters — nothing to enrich

        chapter_reqs = [r for r in report_data.requirements if r.chapter == chapter.chapter]
        needs_llm = [r for r in chapter_reqs if r.verdict != "FULLY_MET"]

        if not needs_llm:
            continue

        print(f"Enriching Chapter {chapter.chapter} ({len(needs_llm)} requirements)...")
        results = enrich_requirements_for_chapter(chapter.name, chapter.article_range, needs_llm)

        for req in chapter_reqs:
            if req.sub_id in results:
                req.analysis = results[req.sub_id]["analysis"]
                req.fix_required = results[req.sub_id]["fix_required"]

    # ---- Step 2: chapter narratives (only weak chapters) ----
    for chapter in report_data.chapters:
        if chapter.total > 0 and needs_narrative(chapter):
            print(f"Writing narrative for Chapter {chapter.chapter}...")
            chapter.narrative = enrich_chapter_narrative(chapter)

    # ---- Step 3: refine priority action wording with real fixes ----
    report_data.priority_actions = refine_priority_actions_with_fixes(
        report_data.priority_actions, report_data.requirements
    )

    # ---- Step 4: executive summary (last, needs full context) ----
    print("Writing executive summary...")
    report_data.executive_summary = generate_executive_summary(report_data)

    return report_data


if __name__ == "__main__":
    from aggregator.aggregator_func import aggregate_results   # your existing function

    with open("sample_report/judge/judge_input.json", "r", encoding="utf-8") as f:
        judge_results = json.load(f)

    aggregated_results = aggregate_results(judge_results)
    report_data = build_report_data(
        aggregated_results,
        meta_overrides={
            "company": "TechStartup Pvt Ltd",
            "policy_analyzed": "Privacy Policy v2.1 (August 2025)",
            "analysis_date": "6 August 2025",
        },
    )

    report_data = enrich_report(report_data)

    output_path = render_report_pdf(report_data, "../output/GDPR_Compliance1.pdf")
    print(f"\nFinal PDF generated: {output_path}")