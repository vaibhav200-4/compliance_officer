"""
chunk_policy.py
-----------------
Step 2: Cleaned policy text ko SECTION-WISE chunk karta hai (word-count-wise nahi).

Kyun section-wise?
  - Policy documents already well-structured hote hain (numbered headings:
    "1 Introduction", "4.2 Inactive Accounts", etc.)
  - Har section ek complete semantic unit hai. Agar hum blind fixed-size
    chunking karein, ek section ka context 2 chunks mein split ho sakta
    hai -> retrieval quality kharab ho jaati hai.
  - Fixed-size chunking sirf tab use karo jab document unstructured ho
    (jaise raw chat logs, ya freeform articles).

Strategy (hybrid):
  1. Regex se numbered headings detect karo (e.g. "4.2 Inactive Accounts")
     aur unke beech ka text ek "section" maano.
  2. Agar koi section chhota hai (< MAX_CHUNK_WORDS) -> ek hi chunk banao.
  3. Agar koi section bada hai (jaise Section 6 ka table, ya Section 4
     jisme kai sub-points hain) -> RecursiveCharacterTextSplitter se
     usko sub-chunks mein todo, with overlap taaki context na tute.

Output: list of dicts, har dict ek chunk hai with metadata.

Usage:
    python chunk_policy.py --txt data/policy_clean.txt --company "TechStartup Pvt Ltd" \
        --out data/policy_chunks.json
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_WORDS = 250   # section is sub-split beyond this
CHUNK_OVERLAP_CHARS = 150

# Matches headings like:
#   "1 Introduction & Scope"
#   "2.1 Information You Provide Directly"
#   "10 Children's Privacy"
SECTION_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})?)\s+(?P<title>[A-Z][A-Za-z0-9 &,'\-]{3,80})$",
    re.MULTILINE,
)


def split_into_sections(text: str):
    """Split cleaned text into (section_number, section_title, section_body) tuples."""
    matches = list(SECTION_HEADING_RE.finditer(text))

    if not matches:
        # Fallback: no headings detected, treat whole doc as one section
        return [("0", "Full Document", text.strip())]

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:  # skip empty (e.g. a false-positive heading match)
            sections.append((m.group("num"), m.group("title").strip(), body))

    return sections


def sub_chunk_if_needed(section_num, section_title, body, splitter):
    """If a section is long, split it further with overlap; else keep as one chunk."""
    word_count = len(body.split())

    if word_count <= MAX_CHUNK_WORDS:
        return [body]

    return splitter.split_text(body)


def build_chunks(text: str, company: str):
    sections = split_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,             # ~ MAX_CHUNK_WORDS in characters
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for section_num, section_title, body in sections:
        sub_texts = sub_chunk_if_needed(section_num, section_title, body, splitter)

        for idx, sub_text in enumerate(sub_texts):
            chunk_id = f"{company.lower().replace(' ', '_')}_sec{section_num}_chunk{idx}"
            # Prepend the section title to the chunk text itself so the
            # embedding "knows" what topic this chunk belongs to even
            # after it's been split out of context. This measurably
            # improves retrieval for policy QA.
            chunk_text = f"Section {section_num} - {section_title}\n{sub_text}"

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "company": company,
                    "section_number": section_num,
                    "section_title": section_title,
                    "chunk_index": idx,
                    "text": chunk_text,  # Pinecone needs text duplicated in metadata for retrieval-time display
                },
            })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Section-wise chunk a cleaned policy text file")
    parser.add_argument("--txt", required=True, help="Path to cleaned .txt (from extract_text.py)")
    parser.add_argument("--company", required=True, help="Company name, used for chunk IDs + metadata")
    parser.add_argument("--out", required=True, help="Path to save chunks as .json")
    args = parser.parse_args()

    text = Path(args.txt).read_text(encoding="utf-8")
    chunks = build_chunks(text, args.company)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created {len(chunks)} chunks from {args.txt}")
    for c in chunks[:3]:
        print(f"  - {c['id']} ({len(c['text'].split())} words)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()