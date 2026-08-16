"""
api/main.py
-----------
FastAPI app exposing your GDPR compliance pipeline.

Run from the compliance_officer/ project root:
    uvicorn api.main:app --reload
"""

import os
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.jobs import job_store
from api.pipeline_runner import run_job
from api.schemas import StatusResponse, UploadResponse

app = FastAPI(title="GDPR Compliance Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload-policy", response_model=UploadResponse)
async def upload_policy(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company: str = Form(...),
    policy_version: str = Form("v1.0"),
    skip_ingest: bool = Form(False),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    job_id = job_store.create_job()

    pdf_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(
        run_job, job_id, str(pdf_path), company, policy_version, skip_ingest
    )

    return UploadResponse(job_id=job_id, message="Pipeline started")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        stage=job["stage"],
        message=job["message"],
        error=job["error"],
    )


@app.get("/download-report/{job_id}")
async def download_report(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready yet (status: {job['status']})",
        )
    result_path = job["result_path"]
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Report file not found on disk")

    return FileResponse(
        result_path,
        filename=os.path.basename(result_path),
        media_type="application/pdf",
    )