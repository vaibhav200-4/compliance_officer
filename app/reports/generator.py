# app/reports/generator.py

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.logger import get_logger

logger = get_logger()

# ----------------------------------------------------------------------
# GDPR CHAPTER MAP
# ----------------------------------------------------------------------
GDPR_CHAPTERS = [
    ("Chapter I", "General Provisions", range(1, 5)),
    ("Chapter II", "Principles", range(5, 12)),
    ("Chapter III", "Rights of the Data Subject", range(12, 24)),
    ("Chapter IV", "Controller and Processor", range(24, 44)),
    ("Chapter V", "Transfers of Personal Data to Third Countries", range(44, 50)),
    ("Chapter VI", "Independent Supervisory Authorities", range(51, 60)),
    ("Chapter VII", "Co-operation and Consistency", range(60, 77)),
    ("Chapter VIII", "Remedies, Liability and Penalties", range(77, 85)),
    ("Chapter IX", "Specific Processing Situations", range(85, 92)),
    ("Chapter X", "Delegated Acts and Implementing Acts", range(92, 94)),
    ("Chapter XI", "Final Provisions", range(94, 100)),
]


class ComplianceReportGenerator:
    """
    Generates professional PDF compliance reports matching the structure
    and visual standard of OutputReport_GDPRCompliance.
    """

    def __init__(self, output_dir: str | Path = "Data/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        analysis_data: dict[str, Any],
        company_name: str = "Target Organization",
        policy_name: str = "Company Privacy Policy",
        output_filename: str | None = None,
    ) -> Path:
        """
        Generate a complete PDF report from structured analysis data atomically.
        """
        t_start = time.perf_counter()
        logger.info(f"PDF_GENERATION_START | company={company_name}")

        if output_filename is None:
            safe_name = "".join(c for c in company_name if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
            output_filename = f"GDPR_Compliance_Report_{safe_name}.pdf"

        final_pdf_path = self.output_dir / output_filename
        temp_pdf_path = final_pdf_path.with_suffix(".tmp.pdf")

        if temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except Exception:
                pass

        # Process and summarize data
        summary = self._compute_summary_data(analysis_data, company_name, policy_name)

        doc = SimpleDocTemplate(
            str(temp_pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#1E293B"),
            alignment=0,
            spaceAfter=6,
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0EA5E9"),
            spaceAfter=15,
        )

        section_heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=12,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )

        table_header = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.white,
        )

        story = []

        # =========================================================================
        # PAGE 1: COVER & EXECUTIVE COMPLIANCE DASHBOARD
        # =========================================================================
        story.append(Paragraph("GDPR READINESS ASSESSMENT", subtitle_style))
        story.append(Paragraph(f"Compliance Audit: {company_name}", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0EA5E9"), spaceAfter=15))

        # Metadata Block Table
        meta_data = [
            [
                Paragraph("<b>Company:</b>", body_style),
                Paragraph(company_name, body_style),
                Paragraph("<b>Audit Standard:</b>", body_style),
                Paragraph("EU GDPR (Regulation 2016/679)", body_style),
            ],
            [
                Paragraph("<b>Policy Document:</b>", body_style),
                Paragraph(policy_name, body_style),
                Paragraph("<b>Audit Date:</b>", body_style),
                Paragraph(datetime.now().strftime("%B %d, %Y"), body_style),
            ],
            [
                Paragraph("<b>Evaluated Groups:</b>", body_style),
                Paragraph(str(summary["total_groups"]), body_style),
                Paragraph("<b>Risk Level:</b>", body_style),
                Paragraph(f"<font color='{summary['risk_color']}'><b>{summary['risk_level']}</b></font>", body_style),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(meta_table)
        story.append(Spacer(1, 15))

        # Score & Status Metrics Grid
        story.append(Paragraph("Overall Compliance Summary", section_heading))

        score_text = f"<font size=26 color='{summary['score_color']}'><b>{summary['overall_score']:.1f}%</b></font><br/><font size=8 color='#64748B'>Compliance Score</font>"
        
        metrics_data = [
            [
                Paragraph(score_text, ParagraphStyle("ScoreCell", parent=body_style, alignment=1)),
                Paragraph(f"<font size=14 color='#10B981'><b>{summary['counts']['MET']}</b></font><br/>Fully Met", ParagraphStyle("MCell", parent=body_style, alignment=1)),
                Paragraph(f"<font size=14 color='#F59E0B'><b>{summary['counts']['PARTIALLY_MET']}</b></font><br/>Partially Met", ParagraphStyle("PMCell", parent=body_style, alignment=1)),
                Paragraph(f"<font size=14 color='#EF4444'><b>{summary['counts']['NOT_MET']}</b></font><br/>Not Met", ParagraphStyle("NMCell", parent=body_style, alignment=1)),
                Paragraph(f"<font size=14 color='#DC2626'><b>{summary['counts']['CONFLICTING']}</b></font><br/>Conflicting", ParagraphStyle("CCell", parent=body_style, alignment=1)),
                Paragraph(f"<font size=14 color='#64748B'><b>{summary['counts']['INSUFFICIENT_EVIDENCE']}</b></font><br/>Insufficient Evidence", ParagraphStyle("IECell", parent=body_style, alignment=1)),
            ]
        ]

        metrics_table = Table(metrics_data, colWidths=[110, 85, 85, 85, 85, 90])
        metrics_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F0F9FF")),
                ("BACKGROUND", (1, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(metrics_table)
        story.append(Spacer(1, 15))

        # Executive Summary Section
        story.append(Paragraph("Executive Overview", section_heading))
        exec_summary_text = (
            f"This GDPR Compliance Audit evaluated <b>{summary['total_groups']} requirement groups</b> "
            f"spanning <b>{summary['total_obligations']} sub-obligations</b> across the General Data Protection Regulation. "
            f"The assessment determined an overall compliance posture of <b>{summary['overall_score']:.1f}%</b> "
            f"with a designated risk classification of <b>{summary['risk_level']}</b>.<br/><br/>"
            f"<b>Key Audit Findings:</b><br/>"
            f"• <b>Fully Satisfied:</b> {summary['counts']['MET']} requirement groups clearly demonstrated compliance.<br/>"
            f"• <b>Partial Gaps:</b> {summary['counts']['PARTIALLY_MET']} requirement groups exhibit partial policy coverage requiring formal procedural additions.<br/>"
            f"• <b>Non-Compliant / Unaddressed:</b> {summary['counts']['NOT_MET']} requirement groups lacked mandatory policy disclosures or operational commitments.<br/>"
            f"• <b>Conflicting Directives:</b> {summary['counts']['CONFLICTING']} groups contain policy language directly contradicting GDPR standards."
        )
        story.append(Paragraph(exec_summary_text, body_style))
        story.append(PageBreak())

        # =========================================================================
        # PAGE 2: CHAPTER-BY-CHAPTER BREAKDOWN
        # =========================================================================
        story.append(Paragraph("Chapter-by-Chapter Compliance Breakdown", section_heading))
        story.append(Paragraph("Structured view of compliance readiness across the 11 GDPR thematic chapters.", body_style))
        story.append(Spacer(1, 10))

        chap_table_data = [
            [
                Paragraph("<b>Chapter</b>", table_header),
                Paragraph("<b>Chapter Name</b>", table_header),
                Paragraph("<b>Articles</b>", table_header),
                Paragraph("<b>Evaluated Groups</b>", table_header),
                Paragraph("<b>Status</b>", table_header),
            ]
        ]

        for chap_code, chap_name, art_range in GDPR_CHAPTERS:
            chap_groups = [g for g in summary["all_groups"] if g["article_number"] in art_range]
            if not chap_groups:
                status_html = "<font color='#94A3B8'>No Direct Obligation Groups</font>"
            else:
                chap_statuses = [g["status"] for g in chap_groups]
                if "CONFLICTING" in chap_statuses:
                    status_html = "<font color='#DC2626'><b>CONFLICTING</b></font>"
                elif "NOT_MET" in chap_statuses:
                    status_html = "<font color='#EF4444'><b>NOT MET</b></font>"
                elif "PARTIALLY_MET" in chap_statuses:
                    status_html = "<font color='#F59E0B'><b>PARTIALLY MET</b></font>"
                elif all(s == "MET" for s in chap_statuses):
                    status_html = "<font color='#10B981'><b>MET</b></font>"
                else:
                    status_html = "<font color='#64748B'><b>INSUFFICIENT EVIDENCE</b></font>"

            art_str = f"Art. {art_range.start}–{art_range.stop - 1}"
            chap_table_data.append([
                Paragraph(f"<b>{chap_code}</b>", body_style),
                Paragraph(chap_name, body_style),
                Paragraph(art_str, body_style),
                Paragraph(str(len(chap_groups)), body_style),
                Paragraph(status_html, body_style),
            ])

        chap_table = Table(chap_table_data, colWidths=[70, 200, 85, 95, 90])
        chap_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ])
        )
        story.append(chap_table)
        story.append(PageBreak())

        # =========================================================================
        # PAGE 3+: DETAILED OBLIGATION & GAP ANALYSIS
        # =========================================================================
        story.append(Paragraph("Detailed Requirement Gap Analysis", section_heading))
        story.append(Paragraph("Granular sub-obligation verdicts grounded in retrieved policy evidence.", body_style))
        story.append(Spacer(1, 10))

        for grp in summary["all_groups"]:
            status_color = self._get_status_color(grp["status"])
            conf_val = grp.get("confidence", 1.0)
            grp_header = (
                f"<b>Group {grp['group_id']} — {grp['principle']}</b> "
                f"(Article {grp['article_number']}) | "
                f"<font color='{status_color}'><b>{grp['status']}</b></font> "
                f"(Confidence: {conf_val * 100:.0f}%)"
            )
            story.append(Paragraph(grp_header, ParagraphStyle("GrpHead", parent=body_style, fontSize=11, leading=14)))

            if grp.get("gap"):
                story.append(Paragraph(f"<b>Identified Gap / Finding:</b> {grp['gap']}", body_style))

            story.append(Spacer(1, 4))

            # Obligations Table
            ob_table_data = [
                [
                    Paragraph("<b>Obligation ID</b>", table_header),
                    Paragraph("<b>Verdict & Confidence</b>", table_header),
                    Paragraph("<b>Reason & Evidence Quote</b>", table_header),
                ]
            ]

            for ob in grp.get("sub_obligations", []):
                ob_status_color = self._get_status_color(ob["status"])
                verdict_html = f"<font color='{ob_status_color}'><b>{ob['status']}</b></font><br/>Conf: {ob.get('confidence', 0.0)*100:.0f}%"

                evidence_text = ""
                for ev in ob.get("evidence", []):
                    evidence_text += f"<br/><i>Quote ({ev.get('chunk_id','')}):</i> \"{ev.get('quote','')}\""

                reason_html = f"<b>Analysis:</b> {ob.get('reason','')}{evidence_text}"

                ob_table_data.append([
                    Paragraph(f"<b>{ob.get('obligation_id','')}</b>", body_style),
                    Paragraph(verdict_html, body_style),
                    Paragraph(reason_html, body_style),
                ])

            ob_table = Table(ob_table_data, colWidths=[90, 110, 340])
            ob_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("PADDING", (0, 0), (-1, -1), 5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )

            story.append(KeepTogether([ob_table, Spacer(1, 10)]))

        story.append(PageBreak())

        # =========================================================================
        # PRIORITY ACTION PLAN & REMEDIATION
        # =========================================================================
        story.append(Paragraph("Priority Action Plan & Remediation Guidance", section_heading))
        story.append(Paragraph("Categorized recommendations based on identified compliance gaps.", body_style))
        story.append(Spacer(1, 10))

        action_table_data = [
            [
                Paragraph("<b>Priority Level</b>", table_header),
                Paragraph("<b>GDPR Article</b>", table_header),
                Paragraph("<b>Group ID</b>", table_header),
                Paragraph("<b>Required Remediation Action</b>", table_header),
            ]
        ]

        priority_count = 0
        for grp in summary["all_groups"]:
            if grp["status"] in {"NOT_MET", "CONFLICTING", "PARTIALLY_MET"}:
                priority_count += 1
                if grp["status"] in {"NOT_MET", "CONFLICTING"}:
                    prio_html = "<font color='#DC2626'><b>P1 CRITICAL</b></font>"
                else:
                    prio_html = "<font color='#F59E0B'><b>P2 HIGH</b></font>"

                action_desc = grp.get("gap") or grp.get("reason") or "Update privacy policy to include required disclosures."

                action_table_data.append([
                    Paragraph(prio_html, body_style),
                    Paragraph(f"Article {grp['article_number']}", body_style),
                    Paragraph(grp["group_id"], body_style),
                    Paragraph(action_desc, body_style),
                ])

        if priority_count == 0:
            action_table_data.append([
                Paragraph("<font color='#10B981'><b>P3 LOW</b></font>", body_style),
                Paragraph("N/A", body_style),
                Paragraph("N/A", body_style),
                Paragraph("No critical policy gaps detected. Maintain current operational compliance.", body_style),
            ])

        action_table = Table(action_table_data, colWidths=[90, 80, 70, 300])
        action_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(action_table)

        # Build PDF Document to temporary file
        doc.build(story)
        t_gen = time.perf_counter() - t_start
        gen_size = temp_pdf_path.stat().st_size if temp_pdf_path.exists() else 0
        logger.info(f"PDF_GENERATION_COMPLETE | path={temp_pdf_path} | size={gen_size} bytes | duration={t_gen:.2f}s")

        # Validate temporary PDF file
        logger.info(f"PDF_VALIDATION_START | path={temp_pdf_path}")
        if not self._validate_pdf_file(temp_pdf_path):
            if temp_pdf_path.exists():
                try:
                    temp_pdf_path.unlink()
                except Exception:
                    pass
            raise ValueError(f"Generated PDF file failed validation checks: {temp_pdf_path}")

        # Atomic replace to final path
        temp_pdf_path.replace(final_pdf_path)
        final_size = final_pdf_path.stat().st_size
        logger.success(f"PDF_VALIDATION_COMPLETE | final_path={final_pdf_path} | size={final_size} bytes")
        return final_pdf_path

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _validate_pdf_file(pdf_path: str | Path) -> bool:
        """
        Validate PDF file header (%PDF-), non-empty size, and %%EOF trailer.
        """
        p = Path(pdf_path)
        if not p.exists():
            logger.error(f"PDF_VALIDATION_FAILED | File does not exist: {p}")
            return False

        size = p.stat().st_size
        if size == 0:
            logger.error(f"PDF_VALIDATION_FAILED | File is 0 bytes: {p}")
            return False

        try:
            with p.open("rb") as f:
                header = f.read(10)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"PDF_VALIDATION_FAILED | Invalid header: {header}")
                    return False

                f.seek(max(0, size - 1024))
                trailer = f.read()
                if b"%%EOF" not in trailer:
                    logger.error("PDF_VALIDATION_FAILED | Missing %%EOF marker in trailer")
                    return False
        except Exception as exc:
            logger.error(f"PDF_VALIDATION_FAILED | File read error: {exc}")
            return False

        logger.info(f"PDF_VALIDATION_COMPLETE | path={p} | size={size} bytes")
        return True

    @staticmethod
    def _compute_summary_data(analysis_data: dict[str, Any], company_name: str, policy_name: str) -> dict[str, Any]:
        """Compute aggregate metrics for report rendering."""
        all_groups = []

        # Handle either full analysis summary or single article dict
        if "article_results" in analysis_data:
            for art_data in analysis_data["article_results"].values():
                art_num = art_data.get("article_number")
                for grp in art_data.get("groups", []):
                    grp_copy = dict(grp)
                    if art_num:
                        grp_copy["article_number"] = art_num
                    elif "group_id" in grp_copy:
                        try:
                            grp_copy["article_number"] = int(grp_copy["group_id"].split(".")[0])
                        except Exception:
                            grp_copy["article_number"] = 0
                    all_groups.append(grp_copy)
        elif "groups" in analysis_data:
            art_num = analysis_data.get("article_number")
            for grp in analysis_data.get("groups", []):
                grp_copy = dict(grp)
                if art_num:
                    grp_copy["article_number"] = art_num
                elif "group_id" in grp_copy:
                    try:
                        grp_copy["article_number"] = int(grp_copy["group_id"].split(".")[0])
                    except Exception:
                        grp_copy["article_number"] = 0
                all_groups.append(grp_copy)
        elif isinstance(analysis_data, dict):
            for k, v in analysis_data.items():
                if isinstance(v, dict) and "groups" in v:
                    art_num = v.get("article_number")
                    for grp in v.get("groups", []):
                        grp_copy = dict(grp)
                        if art_num:
                            grp_copy["article_number"] = art_num
                        elif "group_id" in grp_copy:
                            try:
                                grp_copy["article_number"] = int(grp_copy["group_id"].split(".")[0])
                            except Exception:
                                grp_copy["article_number"] = 0
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
            status = grp.get("status", "INSUFFICIENT_EVIDENCE")
            counts[status] = counts.get(status, 0) + 1
            total_obligations += len(grp.get("sub_obligations", []))

        total_groups = len(all_groups)
        applicable_groups = total_groups - counts["NOT_APPLICABLE"]

        if applicable_groups > 0:
            overall_score = ((counts["MET"] * 1.0 + counts["PARTIALLY_MET"] * 0.5) / applicable_groups) * 100.0
        else:
            overall_score = 0.0

        if counts["CONFLICTING"] > 0 or counts["NOT_MET"] > 3 or overall_score < 50.0:
            risk_level = "CRITICAL RISK"
            risk_color = "#DC2626"
        elif counts["NOT_MET"] > 0 or overall_score < 75.0:
            risk_level = "HIGH RISK"
            risk_color = "#EF4444"
        elif overall_score < 90.0:
            risk_level = "MEDIUM RISK"
            risk_color = "#F59E0B"
        else:
            risk_level = "LOW RISK"
            risk_color = "#10B981"

        if overall_score >= 80.0:
            score_color = "#10B981"
        elif overall_score >= 50.0:
            score_color = "#F59E0B"
        else:
            score_color = "#DC2626"

        return {
            "company_name": company_name,
            "policy_name": policy_name,
            "total_groups": total_groups,
            "total_obligations": total_obligations,
            "counts": counts,
            "overall_score": round(overall_score, 1),
            "score_color": score_color,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "all_groups": all_groups,
        }

    @staticmethod
    def _get_status_color(status: str) -> str:
        mapping = {
            "MET": "#10B981",
            "PARTIALLY_MET": "#F59E0B",
            "NOT_MET": "#EF4444",
            "CONFLICTING": "#DC2626",
            "INSUFFICIENT_EVIDENCE": "#64748B",
            "NOT_APPLICABLE": "#94A3B8",
        }
        return mapping.get(status, "#64748B")
