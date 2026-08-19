#!/usr/bin/env python3
"""
GDPR Compliance Analyzer Application Launcher.
Runs both the FastAPI backend server (port 8000) and the Streamlit frontend (port 8501).
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable

def main():
    print("=" * 75)
    print("STARTING GDPR COMPLIANCE ANALYZER PLATFORM")
    print("=" * 75)

    print("1. Launching FastAPI Backend Server (http://127.0.0.1:8000)...")
    backend_proc = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ],
        cwd=str(PROJECT_ROOT),
    )

    time.sleep(2)

    print("2. Launching Streamlit Frontend (http://127.0.0.1:8501)...")
    frontend_proc = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            "8501",
        ],
        cwd=str(PROJECT_ROOT),
    )

    print()
    print("=" * 75)
    print("SERVICES ARE RUNNING:")
    print("  • FastAPI API Backend : http://127.0.0.1:8000")
    print("  • API Health Endpoint  : http://127.0.0.1:8000/api/health")
    print("  • Streamlit Web App    : http://127.0.0.1:8501")
    print("=" * 75)
    print("Press Ctrl+C to stop all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("All services stopped.")

if __name__ == "__main__":
    main()
