# app/api/routes/policy.py

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.core.config import get_settings
from app.core.logger import get_logger
from app.reports.generator import ComplianceReportGenerator

logger = get_logger()
router = APIRouter()

# Global Thread-Safe Job Store
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

UPLOAD_DIR = Path("Data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _get_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis job '{job_id}' not found.",
            )
        return JOBS[job_id]


def _update_job(job_id: str, **kwargs: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _run_background_analysis(
    job_id: str,
    file_path: Path | None,
    company_name: str,
    policy_name: str,
    start_article: int,
    end_article: int | None,
) -> None:
    """
    Background worker function that executes policy ingestion (if file uploaded),
    runs GDPR compliance analysis, and generates the PDF report.
    """
    start_time = time.time()
    _update_job(
        job_id,
        status="RUNNING",
        message="Job initialized. Ingesting policy if necessary...",
    )

    try:
        # 1. Optional policy ingestion
        if file_path and file_path.exists():
            logger.info(f"[{job_id}] Ingesting uploaded policy: {file_path}")
            try:
                from app.ingestion.pipeline import IngestionPipeline
                pipeline = IngestionPipeline()
                pipeline.ingest(str(file_path))
                logger.success(f"[{job_id}] Policy ingested successfully.")
            except Exception as exc:
                logger.warning(f"[{job_id}] Policy ingestion warning: {exc}; proceeding with existing store.")

        # 2. Compliance Analysis
        logger.info(f"[{job_id}] Starting RAG analysis for {company_name}...")
        kb = GDPRKnowledgeBase()
        orchestrator = ComplianceOrchestrator(knowledge_base=kb)

        tot_requested = (end_article or kb.article_count()) - start_article + 1

        def on_progress(completed_arts: int, total_arts: int, current_art: int, group_id: str) -> None:
            pct = round((completed_arts / max(1, total_arts)) * 100.0, 1)
            _update_job(
                job_id,
                completed_articles=completed_arts,
                total_articles=total_arts,
                current_article=current_art,
                current_group=group_id,
                progress_percent=pct,
                progress=pct,
                message=f"Processing Article {current_art} (Group {group_id}) [{completed_arts}/{total_arts}]",
            )

        logger.info(f"[{job_id}] ANALYSIS_START | articles {start_article}-{end_article}")
        t_anal_start = time.perf_counter()
        result = orchestrator.run(
            start_article=start_article,
            end_article=end_article,
            resume=True,
            company_name=company_name,
            policy_name=policy_name,
            progress_callback=on_progress,
        )
        t_anal_dur = time.perf_counter() - t_anal_start
        logger.info(f"[{job_id}] ANALYSIS_COMPLETE | duration={t_anal_dur:.2f}s")

        logger.info(f"[{job_id}] REPORT_AGGREGATION_START")
        t_rep_start = time.perf_counter()
        final_report = result.get("final_report")
        t_rep_dur = time.perf_counter() - t_rep_start
        logger.info(f"[{job_id}] REPORT_AGGREGATION_COMPLETE | duration={t_rep_dur:.2f}s")

        # 3. PDF Report Generation
        logger.info(f"[{job_id}] PDF_GENERATION_START")
        t_pdf_start = time.perf_counter()
        pdf_path = None
        try:
            report_gen = ComplianceReportGenerator()
            pdf_path = report_gen.generate_pdf_report(
                analysis_data=result,
                company_name=company_name,
                policy_name=policy_name,
                output_filename=f"{job_id}_report.pdf",
            )
            t_pdf_dur = time.perf_counter() - t_pdf_start
            logger.info(f"[{job_id}] PDF_GENERATION_COMPLETE | duration={t_pdf_dur:.2f}s")
        except Exception as exc:
            logger.error(f"[{job_id}] PDF_GENERATION_FAILED | error={exc}")

        elapsed = time.time() - start_time
        final_status = "COMPLETED" if pdf_path and pdf_path.exists() else "PARTIAL"
        logger.info(f"[{job_id}] TOTAL_JOB_TIME | total={elapsed:.2f}s")

        _update_job(
            job_id,
            status=final_status,
            progress_percent=100.0,
            progress=100,
            completed_articles=tot_requested,
            elapsed_seconds=round(elapsed, 1),
            result=result,
            report=final_report,
            report_path=str(pdf_path) if pdf_path else None,
            report_ready=bool(pdf_path and pdf_path.exists()),
            message=f"Job finished with status {final_status}.",
        )
        logger.success(f"[{job_id}] Job finished in {elapsed:.1f}s with status={final_status}.")

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Job failed with exception: {exc}")
        _update_job(
            job_id,
            status="FAILED",
            error=str(exc),
            elapsed_seconds=round(elapsed, 1),
            message=f"Job failed: {exc}",
            report_ready=False,
        )


@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    company_name: str = Form("Target Organization"),
    policy_name: str = Form("Privacy Policy Document"),
    start_article: int = Form(1),
    end_article: int | None = Form(None),
) -> Dict[str, Any]:
    """
    Start background GDPR compliance analysis.
    Returns immediately with a job_id for status polling.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    saved_path: Path | None = None
    if file and file.filename:
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (".", "_", "-"))
        saved_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "company_name": company_name,
            "policy_name": policy_name,
            "file_path": str(saved_path) if saved_path else None,
            "progress": 0,
            "progress_percent": 0.0,
            "current_article": start_article,
            "completed_articles": 0,
            "total_articles": (end_article or 99) - start_article + 1,
            "current_group": "",
            "elapsed_seconds": 0,
            "provider": os.getenv("LLM_PROVIDER", "auto"),
            "message": "Job queued.",
            "result": None,
            "report": None,
            "report_path": None,
            "report_ready": False,
            "error": None,
            "created_at": time.time(),
        }

    background_tasks.add_task(
        _run_background_analysis,
        job_id=job_id,
        file_path=saved_path,
        company_name=company_name,
        policy_name=policy_name,
        start_article=start_article,
        end_article=end_article,
    )

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": "Analysis started in background.",
    }


@router.get("/analyze/{job_id}/status")
async def get_analysis_status(job_id: str) -> Dict[str, Any]:
    """Get current status and progress of an analysis job."""
    job = _get_job(job_id)
    now = time.time()
    raw_status = str(job.get("status", "QUEUED")).upper()
    elapsed = round(now - job["created_at"], 1) if raw_status == "RUNNING" else job.get("elapsed_seconds", 0)

    return {
        "job_id": job["job_id"],
        "status": raw_status,
        "stage": job.get("stage", "ANALYZING"),
        "current_article": job.get("current_article", 1),
        "total_articles": job.get("total_articles", 99),
        "current_group": job.get("current_group", ""),
        "completed_groups": job.get("completed_groups", 0),
        "total_groups": job.get("total_groups", 0),
        "completed_articles": job.get("completed_articles", 0),
        "progress_percent": job.get("progress_percent", 0.0),
        "progress": job.get("progress", 0),
        "elapsed_seconds": elapsed,
        "provider": job.get("provider", "gemini"),
        "message": job.get("message", ""),
        "company_name": job.get("company_name"),
        "error": job.get("error"),
        "report_ready": job.get("report_ready", False),
        "report": job.get("report"),
        "report_path": job.get("report_path"),
    }


@router.get("/analyze/{job_id}/result")
async def get_analysis_result(job_id: str) -> Dict[str, Any]:
    """Get structured JSON result of a completed analysis job."""
    job = _get_job(job_id)

    if str(job["status"]).lower() == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {job.get('error')}",
        )

    if str(job["status"]).lower() not in ["completed", "completed_with_failures"] or not job.get("result"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is currently {job['status']}; results not ready.",
        )

    return job.get("report") or job["result"]


@router.get("/analyze/{job_id}/report")
@router.get("/analyze/{job_id}/download")
async def download_analysis_report(job_id: str) -> FileResponse:
    """Download the generated PDF compliance report."""
    job = _get_job(job_id)
    job_status = str(job.get("status", "")).upper()

    if job_status not in ("COMPLETED", "PARTIAL") or not job.get("report_path"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job status is '{job.get('status')}'; PDF report is not ready.",
        )

    report_path = Path(job["report_path"])
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generated PDF report file not found on server.",
        )

    if report_path.stat().st_size == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF report file is empty or corrupted.",
        )

    company_safe = "".join(c for c in job.get("company_name", "Organization") if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    download_filename = f"{company_safe}_GDPR_Report.pdf"

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=download_filename,
    )
