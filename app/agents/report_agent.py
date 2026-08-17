# app/agents/report_agent.py

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logger import get_logger

logger = get_logger()


class ReportAgent:
    """
    Incremental Parallel Consumer & Aggregator for GDPR compliance reports.

    Responsibilities:
    - Consume validated group/article results incrementally while Analyzer processes later articles.
    - Maintain idempotent internal state under threading lock.
    - Calculate aggregate statistics (MET, PARTIALLY_MET, NOT_MET, CONFLICTING, INSUFFICIENT_EVIDENCE, NOT_APPLICABLE).
    - Finalize structured JSON (`final_report.json`, `report.json`) and Markdown (`report.md`).
    """

    def __init__(self, output_dir: str | Path = "Data/analysis_results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._consumed_groups: dict[str, dict[str, Any]] = {}
        self._article_map: dict[int, dict[str, Any]] = {}
        self._last_article: int = 0
        self._last_group: str = ""
        msg_init = f"REPORT_AGENT_INITIALIZED | Output dir: {self.output_dir}"
        logger.info(msg_init)
        print(msg_init, flush=True)

    def consume_result(self, result_item: dict[str, Any]) -> None:
        """
        Idempotently consume one group or article result item as it finishes.
        Updates in-memory article and group maps safely under lock.
        """
        if not result_item or not isinstance(result_item, dict):
            return

        with self._lock:
            # Check if this is an article-level result
            art_num = result_item.get("article_number")
            groups = result_item.get("groups")

            if art_num is not None and groups is not None and isinstance(groups, list):
                art_int = int(art_num)
                self._last_article = art_int

                if art_int not in self._article_map:
                    self._article_map[art_int] = {
                        "article_number": art_int,
                        "article_title": result_item.get("article_title", f"Article {art_int}"),
                        "status": result_item.get("status", "INSUFFICIENT_EVIDENCE"),
                        "confidence": result_item.get("confidence", 0.0),
                        "groups": {},
                    }
                else:
                    self._article_map[art_int]["article_title"] = result_item.get(
                        "article_title", self._article_map[art_int].get("article_title")
                    )
                    self._article_map[art_int]["status"] = result_item.get(
                        "status", self._article_map[art_int].get("status")
                    )
                    self._article_map[art_int]["confidence"] = result_item.get(
                        "confidence", self._article_map[art_int].get("confidence")
                    )

                for grp in groups:
                    grp_id = grp.get("group_id", "unknown")
                    key = f"{art_int}:{grp_id}"
                    self._last_group = grp_id
                    grp_copy = dict(grp)
                    grp_copy["article_number"] = art_int
                    self._consumed_groups[key] = grp_copy
                    self._article_map[art_int]["groups"][grp_id] = grp_copy

                msg_art = f"REPORT_AGENT_CONSUME_RESULT | Article={art_int} | Total groups={len(groups)}"
                logger.info(msg_art)
                print(msg_art, flush=True)

            elif "group_id" in result_item:
                grp_id = str(result_item["group_id"])
                art_num_val = result_item.get("article_number", 0)
                art_int = int(art_num_val) if art_num_val else 0
                key = f"{art_int}:{grp_id}"

                if key in self._consumed_groups:
                    msg_dup = f"REPORT_AGENT_DUPLICATE_RESULT | key={key}"
                    logger.info(msg_dup)
                    print(msg_dup, flush=True)
                    return

                self._last_article = art_int
                self._last_group = grp_id

                grp_copy = dict(result_item)
                grp_copy["article_number"] = art_int
                self._consumed_groups[key] = grp_copy

                if art_int not in self._article_map:
                    self._article_map[art_int] = {
                        "article_number": art_int,
                        "article_title": f"Article {art_int}",
                        "status": "INSUFFICIENT_EVIDENCE",
                        "confidence": 0.0,
                        "groups": {},
                    }

                self._article_map[art_int]["groups"][grp_id] = grp_copy

                msg_c = f"REPORT_AGENT_CONSUME_RESULT | Article={art_int} | Group={grp_id}"
                logger.info(msg_c)
                print(msg_c, flush=True)

            total_consumed = len(self._consumed_groups)
            msg_upd = f"REPORT_AGENT_UPDATE | Article={self._last_article} | Group={self._last_group} | Total consumed groups={total_consumed}"
            logger.info(msg_upd)

    def get_progress(self) -> dict[str, Any]:
        """Return current progress summary of consumed results."""
        with self._lock:
            return {
                "articles_received": len(self._article_map),
                "groups_received": len(self._consumed_groups),
                "last_article": self._last_article,
                "last_group": self._last_group,
            }

    def finalize_report(
        self,
        company_name: str = "Target Organization",
        policy_name: str = "Company Privacy Policy",
    ) -> dict[str, Any]:
        """
        Finalize and persist complete report after ALL_ARTICLES_COMPLETE.
        Merges all consumed groups into the complete article map.
        """
        msg_fin = "REPORT_AGENT_FINALIZE | Finalizing compliance report..."
        logger.info(msg_fin)
        print(msg_fin, flush=True)

        with self._lock:
            merged_article_map: dict[int, dict[str, Any]] = {}
            for art_num, art_data in self._article_map.items():
                groups_dict = art_data.get("groups", {})
                ordered_groups = list(groups_dict.values()) if isinstance(groups_dict, dict) else art_data.get("groups", [])

                merged_article_map[art_num] = {
                    "article_number": art_num,
                    "article_title": art_data.get("article_title", f"Article {art_num}"),
                    "status": art_data.get("status", "INSUFFICIENT_EVIDENCE"),
                    "confidence": art_data.get("confidence", 0.0),
                    "groups": ordered_groups,
                }

            total_articles_cnt = len(merged_article_map)
            total_groups_cnt = len(self._consumed_groups)

        msg_data = f"REPORT_AGENT_DATA | articles={total_articles_cnt} | groups={total_groups_cnt}"
        logger.info(msg_data)
        print(msg_data, flush=True)

        return self.generate_report(
            analysis_results=merged_article_map,
            company_name=company_name,
            policy_name=policy_name,
        )

    def generate_report(
        self,
        analysis_results: dict[str, Any] | list[dict[str, Any]],
        company_name: str = "Target Organization",
        policy_name: str = "Company Privacy Policy",
    ) -> dict[str, Any]:
        """
        Generate final compliance report from article analysis results.
        """
        logger.info("REPORT_AGENT_START | Aggregating article analysis into final report...")
        logger.info("[REPORT] Article analysis completed.")

        # Standardize input into article map
        article_map: dict[int, dict[str, Any]] = {}
        if isinstance(analysis_results, list):
            for res in analysis_results:
                art_num = res.get("article_number")
                if art_num is not None:
                    article_map[int(art_num)] = res
        elif isinstance(analysis_results, dict):
            if "article_results" in analysis_results and isinstance(analysis_results["article_results"], dict):
                for k, v in analysis_results["article_results"].items():
                    try:
                        article_map[int(k)] = v
                    except ValueError:
                        pass
            else:
                for k, v in analysis_results.items():
                    if isinstance(v, dict) and "article_number" in v:
                        try:
                            article_map[int(k)] = v
                        except ValueError:
                            pass

        logger.info(f"[REPORT] Collected {len(article_map)} article result(s).")
        logger.info("[REPORT] Starting ReportAgent...")

        articles_analyzed = sorted(list(article_map.keys()))
        all_groups: list[dict[str, Any]] = []

        for art_num in articles_analyzed:
            art_data = article_map[art_num]
            for grp in art_data.get("groups", []):
                grp_copy = dict(grp)
                grp_copy["article_number"] = art_num
                all_groups.append(grp_copy)

        counts = {
            "MET": 0,
            "PARTIALLY_MET": 0,
            "NOT_MET": 0,
            "CONFLICTING": 0,
            "INSUFFICIENT_EVIDENCE": 0,
            "NOT_APPLICABLE": 0,
        }

        total_obligations = 0

        for grp in all_groups:
            st_val = grp.get("status", "INSUFFICIENT_EVIDENCE")
            counts[st_val] = counts.get(st_val, 0) + 1
            total_obligations += len(grp.get("sub_obligations", []))

        total_groups = len(all_groups)
        applicable_groups = total_groups - counts["NOT_APPLICABLE"]

        if applicable_groups > 0:
            overall_score = ((counts["MET"] * 1.0 + counts["PARTIALLY_MET"] * 0.5) / applicable_groups) * 100.0
        else:
            overall_score = 0.0

        if counts["CONFLICTING"] > 0 or counts["NOT_MET"] > 0:
            overall_status = "NOT_MET"
        elif counts["PARTIALLY_MET"] > 0:
            overall_status = "PARTIALLY_MET"
        elif counts["MET"] > 0:
            overall_status = "MET"
        else:
            overall_status = "INSUFFICIENT_EVIDENCE"

        if counts["CONFLICTING"] > 0 or counts["NOT_MET"] > 3 or overall_score < 50.0:
            risk_level = "CRITICAL RISK"
        elif counts["NOT_MET"] > 0 or overall_score < 75.0:
            risk_level = "HIGH RISK"
        elif overall_score < 90.0:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        # Executive Summary
        exec_summary = (
            f"GDPR Compliance Audit for '{company_name}' evaluating document '{policy_name}'. "
            f"The evaluation analyzed {len(articles_analyzed)} article(s) spanning {total_groups} requirement group(s) "
            f"and {total_obligations} sub-obligation(s). "
            f"The overall compliance score is {overall_score:.1f}% with an overall status of '{overall_status}' "
            f"and a risk posture of '{risk_level}'."
        )

        # Key Compliance Gaps & Recommendations
        key_gaps: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        for grp in all_groups:
            grp_st = grp.get("status", "INSUFFICIENT_EVIDENCE")
            if grp_st in {"NOT_MET", "CONFLICTING", "PARTIALLY_MET"}:
                gap_desc = grp.get("gap") or grp.get("reason") or "Missing or incomplete policy evidence."
                prio = "CRITICAL" if grp_st in {"NOT_MET", "CONFLICTING"} else "HIGH"
                
                key_gaps.append({
                    "group_id": grp.get("group_id"),
                    "article_number": grp.get("article_number"),
                    "principle": grp.get("principle"),
                    "status": grp_st,
                    "gap": gap_desc,
                })

                recommendations.append({
                    "priority": prio,
                    "group_id": grp.get("group_id"),
                    "article_number": grp.get("article_number"),
                    "action": f"Update policy to fulfill requirements for Group {grp.get('group_id')} ({grp.get('principle')}): {gap_desc}",
                })

        # Final Structured Report Dictionary
        report: dict[str, Any] = {
            "company_name": company_name,
            "policy_name": policy_name,
            "timestamp": datetime.now().isoformat(),
            "status": "COMPLETED",
            "overall_status": overall_status,
            "overall_score": round(overall_score, 1),
            "risk_level": risk_level,
            "articles_analyzed": articles_analyzed,
            "total_groups": total_groups,
            "total_obligations": total_obligations,
            "statistics": counts,
            "executive_summary": exec_summary,
            "key_gaps": key_gaps,
            "recommendations": recommendations,
            "article_results": {str(k): v for k, v in article_map.items()},
            "all_groups": all_groups,
        }

        # 1. Save JSON Report Files
        final_report_json = self.output_dir / "final_report.json"
        report_json = self.output_dir / "report.json"

        for json_path in [final_report_json, report_json]:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        # 2. Save Markdown Report File
        report_md_path = self.output_dir / "report.md"
        markdown_content = self._build_markdown_report(report)
        with report_md_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.success("REPORT_AGENT_COMPLETE | Final compliance report generated.")
        logger.info(f"[REPORT] Saved report -> {final_report_json}")
        logger.info(f"[REPORT] Saved report -> {report_md_path}")
        return report

    def _build_markdown_report(self, report: dict[str, Any]) -> str:
        """Build clean Markdown representation of the compliance report."""
        lines = [
            f"# GDPR Compliance Report — {report['company_name']}",
            "",
            f"**Policy Document:** {report['policy_name']}  ",
            f"**Audit Date:** {report['timestamp']}  ",
            f"**Overall Compliance Status:** `{report['overall_status']}`  ",
            f"**Compliance Score:** `{report['overall_score']}%`  ",
            f"**Risk Level:** `{report['risk_level']}`",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            report["executive_summary"],
            "",
            "## Compliance Statistics",
            "",
            "| Metric | Count |",
            "| :--- | :--- |",
            f"| Fully Met (MET) | {report['statistics'].get('MET', 0)} |",
            f"| Partially Met (PARTIALLY_MET) | {report['statistics'].get('PARTIALLY_MET', 0)} |",
            f"| Not Met (NOT_MET) | {report['statistics'].get('NOT_MET', 0)} |",
            f"| Conflicting (CONFLICTING) | {report['statistics'].get('CONFLICTING', 0)} |",
            f"| Insufficient Evidence | {report['statistics'].get('INSUFFICIENT_EVIDENCE', 0)} |",
            f"| Not Applicable | {report['statistics'].get('NOT_APPLICABLE', 0)} |",
            "",
            "## Key Compliance Gaps",
            "",
        ]

        if not report["key_gaps"]:
            lines.append("No critical compliance gaps detected.")
        else:
            for gap in report["key_gaps"]:
                lines.append(
                    f"- **Group {gap['group_id']}** (Article {gap['article_number']} — {gap['principle']}): "
                    f"Status `{gap['status']}`. Gap: {gap['gap']}"
                )

        lines.extend([
            "",
            "## Priority Recommendations",
            "",
        ])

        if not report["recommendations"]:
            lines.append("No immediate remediation steps required.")
        else:
            for rec in report["recommendations"]:
                lines.append(f"1. **[{rec['priority']}]** (Article {rec['article_number']}): {rec['action']}")

        lines.extend([
            "",
            "---",
            "*Report generated by Compliance Officer RAG Agent*",
        ])

        return "\n".join(lines)
