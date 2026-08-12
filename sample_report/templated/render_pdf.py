"""
render_pdf_playwright.py
Same job as render_pdf.py (Jinja2 -> PDF) but using headless Chromium via Playwright
instead of WeasyPrint. Use this if WeasyPrint's GTK/pango DLLs keep conflicting on
your Windows machine (e.g. with SWI-Prolog's libgobject-2.0-0.dll on PATH).

Setup (one-time):
    pip install playwright --break-system-packages
    playwright install chromium

Usage: identical signature to render_pdf.render_report_pdf, so run_track1.py only
needs a one-line import change:
    from render_pdf_playwright import render_report_pdf
"""

import os
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

from sample_report.templated.models import ReportData

TEMPLATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "templates")
)

def render_report_pdf(data: ReportData, output_path: str) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("report_template.html")
    html_str = template.render(data=data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_str, wait_until="networkidle")
        page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "22mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()

    return output_path