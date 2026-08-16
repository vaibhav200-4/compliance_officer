"""
api/pipeline_runner.py
-----------------------
Bridges FastAPI's BackgroundTasks to your existing run_full_pipeline()
in main_pipeline_test.py.
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from main_pipeline import run_full_pipeline  # noqa: E402
from api.jobs import job_store  # noqa: E402


def run_job(job_id: str, pdf_path: str, company: str, policy_version: str, skip_ingest: bool):
    job_store.update(job_id, status="running", stage="starting", message="Pipeline starting")

    def status_callback(stage: str, message: str):
        job_store.update(job_id, stage=stage, message=message)

    try:
        output_path = run_full_pipeline(
            pdf_path=pdf_path,
            company=company,
            policy_version=policy_version,
            skip_ingest=skip_ingest,
            status_callback=status_callback,
        )
        job_store.update(
            job_id,
            status="done",
            stage="done",
            message="Report ready",
            result_path=output_path,
        )
    except Exception as e:
        traceback.print_exc()
        job_store.update(
            job_id,
            status="failed",
            message=f"Failed: {e}",
            error=str(e),
        )
    finally:
        try:
            if pdf_path and os.path.exists(pdf_path) and not skip_ingest:
                os.remove(pdf_path)
        except OSError:
            pass