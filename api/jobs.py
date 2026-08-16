"""
api/jobs.py
-----------
Simple in-memory job store. Tracks status/stage/result for each pipeline run
by job_id. Lost on server restart -- fine for a demo/single-process setup.
"""

import threading
import time
import uuid
from typing import Dict, Optional


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",       # queued | running | done | failed
                "stage": None,
                "message": "Job queued",
                "result_path": None,
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        return job_id

    def update(self, job_id: str, **fields):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)
                self._jobs[job_id]["updated_at"] = time.time()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


job_store = JobStore()