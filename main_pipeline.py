"""
main_pipeline.py
--------------------
Production end-to-end pipeline: company privacy policy PDF -> final compliance PDF report.

Run from compliance_officer/ root:  python main_pipeline.py --pdf path/to/policy.pdf --company "Acme Corp"


"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

# ---- Known-good imports (Stages 5-8) ----
from aggregator.aggregator_func import aggregate_results
from report.templated.build_report_data import build_report_data
from report.templated.priority_engine import refine_priority_actions_with_fixes
from report.llm_called.llm_client import get_llm
from report.llm_called.llm_schemas import JudgeVerdictBatchOutput
from report.llm_called.enrich_chapters import needs_narrative, enrich_chapter_narrative
from report.llm_called.enrich_requirements import enrich_requirements_for_chapter
from report.llm_called.enrich_executive_summary import generate_executive_summary

# ---- Stages 1-4 and 9: now wired to your actual files ----
import os
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv

from loading.extract_text import extract_raw_text, clean_text
from chunking.policy_chunking import build_chunks
from embedding.embeddings import generate_embeddings, MODEL_NAME as POLICY_EMBEDDING_MODEL
from report.templated.render_pdf import render_report_pdf

load_dotenv()

# SECURITY: never hardcode this. Rotate the key you shared earlier, then put the
# new one in a .env file (same folder as this script) as: PINECONE_API_KEY=your_new_key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "generic-rag"
ARTICLE_EMBEDDINGS_PATH = "dataa/article_embeddings1.npy"
ARTICLE_METADATA_PATH = "dataa/article_metadata1.json"
GDPR_EMBEDDING_DIM = 1024  # must match whatever model embedded article_embeddings1.npy -- confirm this matches POLICY_EMBEDDING_MODEL's output dim (jina-embeddings-v3 = 1024, so this lines up)


# ============================================================
# CONFIG
# ============================================================
PRE_FILTER_THRESHOLD = 0.30
JUDGE_BATCH_SIZE = 6          # drop to 4 if using a smaller/less reliable model
MAX_WORKERS = 5
MAX_REQUESTS_PER_MINUTE = 25  # keep below your actual Groq RPM limit
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2
MIN_SIMILARITY_FOR_EVIDENCE = 0.3

OUTPUT_DIR = Path("sample_report/output")


# ============================================================
# STAGE 1-4: input -> evidence_for_obligations  (# ADAPT)
# ============================================================

def upsert_policy_chunks(embedded_chunks: list[dict], index, batch_size: int = 100):
    """
    Upserts embedded policy chunks into Pinecone. Each chunk dict has
    id / text / metadata / values (values added by embeddings.generate_embeddings).
    """
    vectors = [
        {"id": c["id"], "values": c["values"], "metadata": c["metadata"]}
        for c in embedded_chunks
    ]
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    print(f"  Upserted {len(vectors)} policy chunks to Pinecone index '{PINECONE_INDEX_NAME}'.")


def retrieve_evidence_for_obligations(company_name: str, index) -> list[dict]:
    """Refactored from your retriever.py into a reusable function (same logic, unchanged)."""
    obligation_embeddings = np.load(ARTICLE_EMBEDDINGS_PATH)
    with open(ARTICLE_METADATA_PATH, "r", encoding="utf-8") as f:
        obligation_metadata = json.load(f)
    if obligation_embeddings.shape[0] != len(obligation_metadata):
        raise ValueError(
            f"Mismatch: {obligation_embeddings.shape[0]} embeddings but {len(obligation_metadata)} metadata entries."
        )

    dummy_vector = [0.0] * GDPR_EMBEDDING_DIM
    results = index.query(
        vector=dummy_vector,
        top_k=1000,
        filter={"company": {"$eq": company_name}},
        include_values=True,
        include_metadata=True,
    )
    policy_chunks = [
        {
            "chunk_id": m["id"],
            "embedding": np.array(m["values"], dtype=np.float32),
            "text": m["metadata"].get("text", ""),
        }
        for m in results["matches"]
    ]
    print(f"  Fetched {len(policy_chunks)} policy chunks for {company_name}.")
    if not policy_chunks:
        raise RuntimeError(f"No policy chunks found for '{company_name}' -- check upload succeeded.")

    policy_matrix = np.array([c["embedding"] for c in policy_chunks], dtype=np.float32)

    output = []
    for i, obligation_vec in enumerate(obligation_embeddings):
        query_norm = obligation_vec / (np.linalg.norm(obligation_vec) + 1e-10)
        matrix_norm = policy_matrix / (np.linalg.norm(policy_matrix, axis=1, keepdims=True) + 1e-10)
        similarities = matrix_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:3]

        evidence = [
            {
                "chunk_id": policy_chunks[idx]["chunk_id"],
                "text": policy_chunks[idx]["text"],
                "similarity": round(float(similarities[idx]), 4),
            }
            for idx in top_indices
        ]
        obligation_entry = dict(obligation_metadata[i])
        obligation_entry.pop("embedding", None)
        obligation_entry["evidence"] = evidence
        output.append(obligation_entry)

    return output


def run_ingestion_and_retrieval(pdf_path: str, company_name: str) -> list[dict]:
    """Stages 1-4: extract -> chunk -> embed+upload -> retrieve evidence per obligation."""
    print(f"[1/9] Extracting text from {pdf_path}...")
    raw_text = extract_raw_text(pdf_path)
    cleaned_text = clean_text(raw_text)

    print(f"[2/9] Chunking policy text...")
    chunks = build_chunks(cleaned_text, company=company_name)
    print(f"  Created {len(chunks)} chunks.")

    print(f"[3/9] Embedding policy chunks ({POLICY_EMBEDDING_MODEL})...")
    embedded_chunks, dim = generate_embeddings(chunks)
    if dim != GDPR_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: policy chunks embedded at dim={dim}, "
            f"but GDPR_EMBEDDING_DIM={GDPR_EMBEDDING_DIM}. article_embeddings1.npy "
            f"must have been embedded with the SAME model as policy chunks, or "
            f"similarity scores will be meaningless."
        )

    print(f"[3b/9] Uploading to Pinecone...")
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set. Add it to a .env file (see security note above).")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    upsert_policy_chunks(embedded_chunks, index)

    print(f"[4/9] Retrieving evidence per GDPR obligation...")
    evidence_for_obligations = retrieve_evidence_for_obligations(company_name, index)

    return evidence_for_obligations


# ============================================================
# STAGE 5: judge_obligations (pre-filter + batched + parallel + rate-limited)
# ============================================================

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = """You are a GDPR compliance judge. You will be given MULTIPLE GDPR obligations,
each with its own retrieved policy evidence excerpts. Judge EACH ONE independently.

For each obligation, decide:
- FULLY_MET: evidence clearly and completely satisfies the obligation.
- PARTIALLY_MET: evidence addresses the obligation but is incomplete/vague.
- NOT_MET: no relevant evidence, or evidence clearly does not satisfy the obligation.
- CONFLICTING: different evidence excerpts contradict each other on this obligation.

Only use the evidence given. Return exactly one result per obligation_id given, matched by obligation_id."""


class RateLimiter:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.timestamps = deque()
        self.lock = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] > 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_per_minute:
                    self.timestamps.append(now)
                    return
                sleep_time = 60 - (now - self.timestamps[0])
            time.sleep(max(sleep_time, 0.1))


def best_similarity(obligation):
    if not obligation["evidence"]:
        return 0.0
    return max(e["similarity"] for e in obligation["evidence"])


def make_auto_not_met(obligation):
    return {
        "article_number": obligation["article_number"],
        "article_name": obligation["article_name"],
        "chapter_number": obligation["chapter_number"],
        "obligation_id": obligation["id"],
        "severity": obligation.get("severity", "MEDIUM"),
        "verdict": "NOT_MET",
        "confidence": 0.95,
        "reason": "No sufficiently relevant policy text was found for this obligation.",
        "gap": "No matching provision found in the policy.",
        "evidence": [],
    }


def build_batch_prompt(batch):
    blocks = []
    for o in batch:
        evidence_block = "\n".join(
            f"  - (similarity {e['similarity']}): {e['text']}"
            for e in o["evidence"] if e["similarity"] >= MIN_SIMILARITY_FOR_EVIDENCE
        ) or "  (no sufficiently relevant excerpts)"
        blocks.append(f"""obligation_id: {o['id']}
Article {o['article_number']} - {o['article_name']}
Legal text: {o['text']}
Evidence:
{evidence_block}""")
    return "Judge the following obligations:\n\n" + "\n\n---\n\n".join(blocks)


def judge_one_batch(batch, structured_llm, rate_limiter):
    user_prompt = build_batch_prompt(batch)
    backoff = INITIAL_BACKOFF_SECONDS
    batch_ids = {o["id"] for o in batch}
    obligation_by_id = {o["id"]: o for o in batch}

    for attempt in range(1, MAX_RETRIES + 1):
        rate_limiter.wait_for_slot()
        try:
            result: JudgeVerdictBatchOutput = structured_llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", user_prompt),
            ])
            output, returned_ids = [], set()
            for item in result.items:
                if item.obligation_id not in batch_ids:
                    continue
                o = obligation_by_id[item.obligation_id]
                returned_ids.add(item.obligation_id)
                output.append({
                    "article_number": o["article_number"],
                    "article_name": o["article_name"],
                    "chapter_number": o["chapter_number"],
                    "obligation_id": o["id"],
                    "severity": o.get("severity", "MEDIUM"),
                    "verdict": item.verdict,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "gap": item.gap,
                    "evidence": [
                        {"chunk_id": e["chunk_id"], "text": e["text"], "page": None, "similarity": e["similarity"]}
                        for e in o["evidence"] if e["similarity"] >= MIN_SIMILARITY_FOR_EVIDENCE
                    ],
                })
            missing = batch_ids - returned_ids
            if missing:
                raise ValueError(f"LLM omitted {len(missing)} obligation(s) from batch response")
            return output
        except Exception as e:
            is_rate_limit = "429" in str(e).lower() or "rate limit" in str(e).lower()
            if attempt == MAX_RETRIES:
                print(f"  BATCH FAILED (final attempt), {len(batch)} obligations lost: {e}")
                return []
            wait = backoff if is_rate_limit else backoff / 2
            print(f"  retry {attempt}/{MAX_RETRIES} for batch of {len(batch)} "
                  f"({'rate limit' if is_rate_limit else 'error'}), waiting {wait:.1f}s")
            time.sleep(wait)
            backoff *= 2
    return []


def run_judge(evidence_for_obligations: list[dict], checkpoint_path: str = "dataa/judge_input.json") -> list[dict]:
    """Stage 5: pre-filter + batch + parallel + rate-limited LLM judging, with checkpointing."""
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["obligation_id"] for r in results}
        print(f"  Resuming: {len(results)} obligations already judged.")
    except FileNotFoundError:
        results, done_ids = [], set()

    remaining = [o for o in evidence_for_obligations if o["id"] not in done_ids]
    if not remaining:
        return results

    needs_llm = []
    for o in remaining:
        if best_similarity(o) < PRE_FILTER_THRESHOLD:
            results.append(make_auto_not_met(o))
        else:
            needs_llm.append(o)
    print(f"  Pre-filter: {len(remaining) - len(needs_llm)} auto-NOT_MET, {len(needs_llm)} need LLM.")

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if not needs_llm:
        return results

    batches = [needs_llm[i:i + JUDGE_BATCH_SIZE] for i in range(0, len(needs_llm), JUDGE_BATCH_SIZE)]
    print(f"  Batched into {len(batches)} LLM calls.")

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(JudgeVerdictBatchOutput)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(judge_one_batch, b, structured_llm, rate_limiter): b for b in batches}
        completed = 0
        for future in as_completed(futures):
            results.extend(future.result())
            completed += 1
            if completed % 3 == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Judge done: {len(results)}/{len(evidence_for_obligations)} obligations judged.")
    return results


# ============================================================
# STAGE 6-8: aggregate -> build_report_data -> Track B enrich
# ============================================================

def _with_retry(fn, *args, label="", max_retries=3, **kwargs):
    """Generic retry wrapper -- enrich_chapters.py/enrich_requirements.py have no
    internal retry, so a single transient error would otherwise crash the whole run."""
    backoff = 1.5
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                print(f"    {label} FAILED after {max_retries} attempts: {e}")
                return None
            print(f"    retry {attempt}/{max_retries} for {label}...")
            time.sleep(backoff)
            backoff *= 2


ENRICHMENT_MAX_WORKERS = 4  # concurrent chapters -- keep modest, these functions have no internal rate limiting


def run_track_b(report_data):
    chapter_lookup = {c.chapter: c for c in report_data.chapters}

    # --- 1. Chapter narratives (parallel across chapters that need one) ---
    print("\n--- Track B: chapter narratives (parallel) ---")
    chapters_needing_narrative = [c for c in report_data.chapters if needs_narrative(c)]

    with ThreadPoolExecutor(max_workers=ENRICHMENT_MAX_WORKERS) as executor:
        future_to_chapter = {
            executor.submit(_with_retry, enrich_chapter_narrative, c, label=f"Ch{c.chapter} narrative"): c
            for c in chapters_needing_narrative
        }
        for future in as_completed(future_to_chapter):
            chapter = future_to_chapter[future]
            result = future.result()
            if result is not None:
                chapter.narrative = result
                print(f"  Chapter {chapter.chapter} narrative done.")

    # --- 2. Requirement analysis + fixes (parallel across chapters) ---
    print("\n--- Track B: requirement analysis + fixes (parallel across chapters) ---")
    reqs_by_chapter = defaultdict(list)
    for req in report_data.requirements:
        if req.verdict != "FULLY_MET":
            reqs_by_chapter[req.chapter].append(req)

    def _enrich_chapter_requirements(chapter_num, reqs):
        cmeta = chapter_lookup[chapter_num]
        print(f"  Chapter {chapter_num}: starting {len(reqs)} requirement(s)...")
        results = _with_retry(
            enrich_requirements_for_chapter, cmeta.name, cmeta.article_range, reqs,
            label=f"Ch{chapter_num} requirements",
        )
        return chapter_num, (results or {})

    with ThreadPoolExecutor(max_workers=ENRICHMENT_MAX_WORKERS) as executor:
        futures = [
            executor.submit(_enrich_chapter_requirements, chapter_num, reqs)
            for chapter_num, reqs in reqs_by_chapter.items()
        ]
        for future in as_completed(futures):
            chapter_num, results = future.result()
            for req in reqs_by_chapter[chapter_num]:
                if req.sub_id in results:
                    req.analysis = results[req.sub_id]["analysis"]
                    req.fix_required = results[req.sub_id]["fix_required"]
            print(f"  Chapter {chapter_num}: requirement enrichment done.")

    # --- 3. Refine priority actions with the sharper fix_required text ---
    print("\n--- Track B: refining priority actions ---")
    report_data.priority_actions = refine_priority_actions_with_fixes(
        report_data.priority_actions, report_data.requirements
    )

    # --- 4. Executive summary (must run last -- needs everything above filled in) ---
    print("\n--- Track B: executive summary ---")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            report_data.executive_summary = generate_executive_summary(report_data)
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  Executive summary FAILED: {e}")
                report_data.executive_summary = "(generation failed)"
            else:
                print(f"  retry {attempt}/{MAX_RETRIES} for executive summary...")
                time.sleep(1.5)

    return report_data


# ============================================================
# REUSABLE ORCHESTRATOR (called by both CLI main() and api/pipeline_runner.py)
# ============================================================

def run_full_pipeline(
    pdf_path: str,
    company: str,
    policy_version: str = "v1.0",
    skip_ingest: bool = False,
    status_callback=None,
) -> str:
    """
    Runs all 9 pipeline stages end to end and returns the output PDF path.

    status_callback(stage: str, message: str) is called at the start of every
    stage. If None, this behaves like a plain function call (prints only).
    This function raises on failure -- the caller (CLI main() or the FastAPI
    job runner) decides how to handle/report that.

    skip_ingest is accepted for API-layer flexibility but is not exposed via
    the CLI here (main_pipeline.py always ingests fresh). If you want a CLI
    flag for it too, use main_pipeline_test.py instead.
    """

    def status(stage: str, message: str):
        print(f"[{stage}] {message}")
        if status_callback:
            status_callback(stage, message)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status("start", f"Starting pipeline for {company}")

    if skip_ingest:
        status("1-4/9", "Skipping ingestion -- reusing existing Pinecone vectors for this company")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set. Add it to a .env file.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        evidence_for_obligations = retrieve_evidence_for_obligations(company, index)
    else:
        status("1-4/9", "Extracting, chunking, embedding, uploading, retrieving evidence")
        evidence_for_obligations = run_ingestion_and_retrieval(pdf_path, company)

    # Stage 5
    status("5/9", "Judging obligations against evidence (LLM)")
    judge_results = run_judge(evidence_for_obligations)

    # Stage 6
    status("6/9", "Aggregating results")
    aggregated = aggregate_results(judge_results)

    # Stage 7
    status("7/9", "Building report data (Track A, rule-based)")
    report_data = build_report_data(
        aggregated,
        meta_overrides={"company": company, "policy_analyzed": f"Privacy Policy {policy_version}"},
        max_requirements=12,
        max_priority_per_tier=5,
    )

    # Stage 8
    status("8/9", "Running Track B enrichment (LLM narratives + analysis)")
    report_data = run_track_b(report_data)

    # Stage 9
    status("9/9", "Rendering PDF")
    output_path = OUTPUT_DIR / f"{company.replace(' ', '_')}_GDPR_Report.pdf"
    render_report_pdf(report_data, str(output_path))

    # Always save the full ReportData as JSON too, useful for debugging/re-rendering
    with open(OUTPUT_DIR / f"{company.replace(' ', '_')}_report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data.model_dump(), f, indent=2, ensure_ascii=False)

    status("done", f"Report ready: {output_path}")
    return str(output_path)


# ============================================================
# CLI ENTRY POINT (unchanged behaviour -- no --skip-ingest here)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end GDPR compliance report pipeline")
    parser.add_argument("--pdf", required=True, help="Path to company privacy policy PDF")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--policy-version", default="v1.0", help="Label for the policy version analyzed")
    args = parser.parse_args()

    print("=" * 60)
    print(f"GDPR Compliance Pipeline -- {args.company}")
    print("=" * 60)

    output_path = run_full_pipeline(
        pdf_path=args.pdf,
        company=args.company,
        policy_version=args.policy_version,
        skip_ingest=False,
    )

    print(f"\nSaved PDF -> {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()