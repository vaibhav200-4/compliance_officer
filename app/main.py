# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.main import api_router

app = FastAPI(
    title="GDPR Compliance Analyzer API",
    description="AI-powered GDPR Privacy Policy Gap Analysis & Audit System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Verify application availability."""
    return {
        "status": "ok",
        "service": "GDPR Compliance Analyzer",
    }

app.include_router(api_router, prefix="/api")
