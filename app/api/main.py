# app/api/main.py

from fastapi import APIRouter
from app.api.routes import policy

api_router = APIRouter()
api_router.include_router(policy.router, tags=["Compliance Analysis"])
