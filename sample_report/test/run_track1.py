"""
run_track1.py
End-to-end Track A run: judge_input.json -> aggregate_results() -> build_report_data()
-> render_pdf(). No LLM calls anywhere in this file.
"""

import json

from aggregator.aggregator_func import aggregate_results   # your existing function
from sample_report.templated.build_report_data import build_report_data
from sample_report.templated.render_pdf import render_report_pdf

if __name__ == "__main__":
    with open("../templated/judge_input.json", "r", encoding="utf-8") as f:
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

    output_path = render_report_pdf(report_data, "../output/GDPR_Compliance_Report.pdf")
    print(f"PDF generated: {output_path}")
    print(f"Overall score: {report_data.overall.score}%")
    print(f"Priority actions: P1={sum(1 for a in report_data.priority_actions if a.priority=='P1')}, "
          f"P2={sum(1 for a in report_data.priority_actions if a.priority=='P2')}, "
          f"P3={sum(1 for a in report_data.priority_actions if a.priority=='P3')}")