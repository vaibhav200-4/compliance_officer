from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str                 # queued | running | done | failed
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None