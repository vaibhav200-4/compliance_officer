"""
extract_text.py
----------------
Step 1: PDF se raw text nikalta hai aur repeated headers/footers clean karta hai.

Input : path to a policy PDF (e.g. data/SampleInput_CompanyPolicy.pdf)
Output: ek single cleaned text string (poore document ka)

Usage:
    python extract_text.py --pdf data/policy.pdf --out data/policy_clean.txt
"""

import argparse
import re
from pathlib import Path

import pdfplumber


def extract_raw_text(pdf_path: str) -> str:
    """Extract raw text page by page using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)


def clean_text(raw_text: str) -> str:
    """
    Remove repeated page headers/footers and normalize whitespace.

    Company policy PDFs usually repeat a line like:
        "<Company> Privacy Policy v2.1 — August 2025"
        "CONFIDENTIAL · Internal Document   Page 3"
    on every page. We detect and strip these using regex patterns
    rather than hardcoding the company name, so this works for
    ANY company's policy PDF, not just this one.
    """
    lines = raw_text.split("\n")
    cleaned_lines = []

    footer_patterns = [
        r"^CONFIDENTIAL.*Page\s*\d+\s*$",           # "CONFIDENTIAL · Internal Document Page 3"
        r".*Privacy Policy.*\d{4}\s*$",              # "... Privacy Policy v2.1 — August 2025"
        r"^Page\s*\d+\s*$",                          # standalone "Page 3"
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in footer_patterns):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # PDFs sometimes wrap 2-digit section numbers around the heading text
    # due to layout, producing: "1\nChildren's Privacy\n0\n..." instead of
    # "10 Children's Privacy\n...". Detect this digit-title-digit pattern
    # and rejoin it into a normal "10 Children's Privacy" heading line.
    text = re.sub(
        r"\n(\d)\n([A-Z][A-Za-z0-9 &,'\-]{3,60})\n(\d)\n",
        r"\n\1\3 \2\n",
        text,
    )

    # Collapse 3+ newlines into 2 (paragraph breaks), single newlines
    # that are just wrapped sentences get joined with a space.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract + clean text from a policy PDF")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument("--out", required=True, help="Path to save the cleaned .txt output")
    args = parser.parse_args()

    raw = extract_raw_text(args.pdf)
    cleaned = clean_text(raw)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")

    print(f"Extracted {len(raw)} raw chars -> {len(cleaned)} cleaned chars")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()