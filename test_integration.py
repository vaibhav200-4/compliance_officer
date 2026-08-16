#!/usr/bin/env python3
"""
Integration Verification Test for FastAPI app, endpoints, and PDF report generator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.reports.generator import ComplianceReportGenerator


def test_integration():
    print("=" * 75)
    print("RUNNING INTEGRATION VERIFICATION CHECKS")
    print("=" * 75)

    client = TestClient(app)

    # 1. Health Endpoint Test
    print("\n1. Testing GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    health_json = resp.json()
    assert health_json.get("status") == "ok", f"Invalid status: {health_json}"
    print(f"   [PASS] Health check returned 200 OK: {health_json}")

    # 2. PDF Report Generator Test
    print("\n2. Testing PDF Report Generation from Article 5 data...")
    art5_file = PROJECT_ROOT / "Data" / "analysis_results" / "article_5.json"
    assert art5_file.exists(), f"Missing test data file: {art5_file}"

    with art5_file.open("r", encoding="utf-8") as f:
        art5_data = json.load(f)

    report_gen = ComplianceReportGenerator(output_dir=PROJECT_ROOT / "Data" / "reports")
    pdf_path = report_gen.generate_pdf_report(
        analysis_data=art5_data,
        company_name="Acme Global Inc.",
        policy_name="Privacy Policy 2026",
        output_filename="Test_Integration_Report.pdf",
    )

    assert pdf_path.exists(), f"PDF report not created: {pdf_path}"
    assert pdf_path.stat().st_size > 1000, f"PDF file too small: {pdf_path.stat().st_size} bytes"
    print(f"   [PASS] PDF Report generated successfully ({pdf_path.stat().st_size} bytes): {pdf_path}")

    print("\n" + "=" * 75)
    print("ALL INTEGRATION VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(test_integration())
