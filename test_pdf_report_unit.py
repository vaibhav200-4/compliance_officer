# test_pdf_report_unit.py

from pathlib import Path
from app.reports.generator import ComplianceReportGenerator

def test_pdf_report_generation():
    print("PDF_UNIT_TEST_START", flush=True)

    static_analysis_data = {
        "company_name": "Acme Corporation",
        "policy_name": "Global Privacy Policy",
        "status": "COMPLETED",
        "article_results": {
            "14": {
                "article_number": 14,
                "article_title": "Information to be provided where personal data have not been obtained from data subject",
                "status": "PARTIALLY_MET",
                "confidence": 0.85,
                "groups": [
                    {
                        "group_id": "14.1",
                        "article_number": 14,
                        "principle": "Controller identification",
                        "status": "MET",
                        "reason": "Controller identity clearly declared.",
                        "gap": "None",
                        "sub_obligations": []
                    },
                    {
                        "group_id": "14.2",
                        "article_number": 14,
                        "principle": "Purpose of processing",
                        "status": "NOT_MET",
                        "reason": "Purposes not specified.",
                        "gap": "Missing processing purposes.",
                        "sub_obligations": []
                    }
                ]
            }
        }
    }

    generator = ComplianceReportGenerator(output_dir="Data/test_reports")
    pdf_path = generator.generate_pdf_report(
        analysis_data=static_analysis_data,
        company_name="Acme Corporation",
        policy_name="Global Privacy Policy",
        output_filename="test_unit_report.pdf",
    )

    assert pdf_path.exists(), "PDF file must exist."
    assert pdf_path.stat().st_size > 0, "PDF file must be non-empty."

    with pdf_path.open("rb") as f:
        header = f.read(10)
        assert header.startswith(b"%PDF-"), f"Invalid PDF header: {header}"

    print(f"PDF_UNIT_TEST_PASSED | generated valid PDF -> {pdf_path}", flush=True)

if __name__ == "__main__":
    test_pdf_report_generation()
