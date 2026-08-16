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
    _update_job(job_id, status="running")

    try:
        # 1. Option policy ingestion
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

        result = orchestrator.run(
            start_article=start_article,
            end_article=end_article,
            resume=True,
        )

        # 3. PDF Report Generation
        logger.info(f"[{job_id}] Generating PDF compliance report...")
        report_gen = ComplianceReportGenerator()
        pdf_path = report_gen.generate_pdf_report(
            analysis_data=result,
            company_name=company_name,
            policy_name=policy_name,
            output_filename=f"{job_id}_report.pdf",
        )

        elapsed = time.time() - start_time
        _update_job(
            job_id,
            status="completed",
            progress=100,
            elapsed_seconds=round(elapsed, 1),
            result=result,
            report_path=str(pdf_path),
        )
        logger.success(f"[{job_id}] Job completed in {elapsed:.1f}s.")

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception(f"[{job_id}] Job failed: {exc}")
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            elapsed_seconds=round(elapsed, 1),
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
            "status": "queued",
            "company_name": company_name,
            "policy_name": policy_name,
            "file_path": str(saved_path) if saved_path else None,
            "progress": 0,
            "articles_completed": 0,
            "articles_total": 99,
            "elapsed_seconds": 0,
            "result": None,
            "report_path": None,
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
        "status": "queued",
        "message": "Analysis started in background.",
    }


@router.get("/analyze/{job_id}/status")
async def get_analysis_status(job_id: str) -> Dict[str, Any]:
    """Get current status and progress of an analysis job."""
    job = _get_job(job_id)
    now = time.time()
    elapsed = round(now - job["created_at"], 1) if job["status"] == "running" else job.get("elapsed_seconds", 0)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "elapsed_seconds": elapsed,
        "company_name": job.get("company_name"),
        "error": job.get("error"),
    }


@router.get("/analyze/{job_id}/result")
async def get_analysis_result(job_id: str) -> Dict[str, Any]:
    """Get structured JSON result of a completed analysis job."""
    job = _get_job(job_id)

    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {job.get('error')}",
        )

    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is currently {job['status']}; results not ready.",
        )

    return job["result"]


@router.get("/analyze/{job_id}/report")
@router.get("/analyze/{job_id}/download")
async def download_analysis_report(job_id: str) -> FileResponse:
    """Download the generated PDF compliance report."""
    job = _get_job(job_id)

    if job["status"] != "completed" or not job.get("report_path"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is {job['status']}; PDF report is not ready.",
        )

    report_path = Path(job["report_path"])
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_444_NOT_FOUND if hasattr(status, 'HTTP_444_NOT_FOUND') else 404,
            detail="Generated report file not found on server.",
        )

    company_safe = "".join(c for c in job.get("company_name", "Organization") if c.isalnum() or c in (" ", "_")).replace(" ", "_")
    download_filename = f"GDPR_Compliance_Report_{company_safe}.pdf"

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=download_filename,
    )
