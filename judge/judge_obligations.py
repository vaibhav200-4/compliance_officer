"""
judge_obligations_prefilter_batched.py
------------------------------------------
Most efficient version: combines THREE optimizations to minimize LLM calls
while staying safely under Groq's rate limit.

1. PRE-FILTER: obligations whose best evidence similarity is below
   PRE_FILTER_THRESHOLD get an automatic NOT_MET verdict -- no LLM call at all.
   These are cases where retrieval itself found nothing relevant, so asking
   the LLM would just waste a call to confirm what we already know.

2. BATCHING: the remaining (borderline/relevant) obligations are grouped into
   batches of BATCH_SIZE. ONE LLM call judges an entire batch at once via
   structured output (a list), instead of one call per obligation.
   411 obligations -> maybe ~150 pass the filter -> ~25 batches of 6.

3. PARALLEL + RATE-LIMITED: batches (not individual obligations) are processed
   concurrently via ThreadPoolExecutor, throttled by the same RateLimiter
   pattern as before, with retry/backoff and checkpointing.

Input:  evidence_for_obligations.json
Output: judge_input.json
"""

import json
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from report.llm_called.llm_client import get_llm
from report.llm_called.llm_schemas import JudgeVerdictBatchOutput
from applicability.company_profile import extract_company_profile
from applicability.filter_obligations import filter_applicable
# ---------- CONFIG ----------
EVIDENCE_INPUT_PATH = "dataa/evidence_for_obligations.json"
OUTPUT_PATH = "dataa/judge_input.json"

MIN_SIMILARITY_FOR_EVIDENCE = 0.3     # evidence below this isn't shown to the LLM at all
          # obligations whose BEST similarity is below this -> auto NOT_MET, skip LLM entirely
PRE_FILTER_THRESHOLD = 0.30
BATCH_SIZE = 6
MAX_WORKERS = 5
MAX_REQUESTS_PER_MINUTE = 25   # apna actual Groq limit confirm karke set karna
       # set below your actual Groq RPM limit
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2
# -----------------------------

# ---------- APPLICABILITY CONFIG ----------
TAGGED_METADATA_PATH = "dataa/article_metadata1_tagged.json"
POLICY_CHUNKS_PATH = "chunking/policy_chunks.json"
COMPANY_PROFILE_CACHE_PATH = "dataa/company_profile.json"
# --------------------------------------------

SYSTEM_PROMPT = """You are a GDPR compliance judge. You will be given MULTIPLE GDPR obligations,
each with its own retrieved policy evidence excerpts. Judge EACH ONE independently.

For each obligation, decide:
- FULLY_MET: evidence clearly and completely satisfies the obligation.
- PARTIALLY_MET: evidence addresses the obligation but is incomplete/vague.
- NOT_MET: no relevant evidence, or evidence clearly does not satisfy the obligation.
- CONFLICTING: different evidence excerpts contradict each other on this obligation.

Only use the evidence given for each obligation -- do not assume anything not stated.
Return exactly one result per obligation_id given, matched by obligation_id."""


class RateLimiter:
    """Shared token-bucket limiter -- same pattern as before, now guarding batch calls."""
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
    """Pre-filtered result: no LLM call, deterministic NOT_MET."""
    return {
        "article_number": obligation["article_number"],
        "article_name": obligation["article_name"],
        "chapter_number": obligation["chapter_number"],
        "obligation_id": obligation["id"],
        "severity": obligation.get("severity", "MEDIUM"),
        "verdict": "NOT_MET",
        "confidence": 0.95,
        "reason": "No sufficiently relevant policy text was found for this obligation (below similarity threshold).",
        "gap": f"Not applicable — this obligation only applies when: {', '.join(obligation.get('applicability_conditions', []))}.",
        "evidence": [],
    }

def make_not_applicable(obligation):
    """Not applicable to this company's processing activities -- no LLM call, no penalty."""
    return {
        "article_number": obligation["article_number"],
        "article_name": obligation["article_name"],
        "chapter_number": obligation["chapter_number"],
        "obligation_id": obligation["id"],
        "severity": obligation.get("severity", "MEDIUM"),
        "verdict": "NOT_APPLICABLE",
        "confidence": 1.0,
        "reason": f"Not applicable -- requires: {', '.join(obligation.get('applicability_conditions', []))}",
        "gap": f"Not applicable — this obligation only applies when: {', '.join(obligation.get('applicability_conditions', []))}.",
        "evidence": [],
    }


def load_tagged_conditions():
    """id -> applicability_conditions lookup from the tagged metadata file."""
    try:
        with open(TAGGED_METADATA_PATH, "r", encoding="utf-8") as f:
            tagged = json.load(f)
        return {o["id"]: o.get("applicability_conditions", []) for o in tagged}
    except FileNotFoundError:
        print(f"  WARNING: {TAGGED_METADATA_PATH} not found -- treating ALL obligations as universal (no filtering).")
        return {}


def get_or_extract_company_profile():
    """Cache the profile so a resumed/checkpointed run doesn't re-extract it."""
    try:
        with open(COMPANY_PROFILE_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print("  Using cached company profile.")
        return cached
    except FileNotFoundError:
        pass

    with open(POLICY_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    full_text = "\n".join(c["text"] for c in chunks)

    profile = extract_company_profile(full_text)
    profile_dict = profile.dict() if hasattr(profile, "dict") else profile
    with open(COMPANY_PROFILE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile_dict, f, indent=2)
    return profile_dict

def build_batch_prompt(batch):
    blocks = []
    for obligation in batch:
        evidence_block = "\n".join(
            f"  - (similarity {e['similarity']}): {e['text']}"
            for e in obligation["evidence"]
            if e["similarity"] >= MIN_SIMILARITY_FOR_EVIDENCE
        )
        if not evidence_block:
            evidence_block = "  (no sufficiently relevant excerpts)"

        blocks.append(f"""obligation_id: {obligation['id']}
Article {obligation['article_number']} - {obligation['article_name']}
Section: {obligation.get('section_name', '')}
Legal text: {obligation['text']}
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

            output = []
            returned_ids = set()
            for item in result.items:
                if item.obligation_id not in batch_ids:
                    continue  # LLM hallucinated an id not in this batch -- skip it
                obligation = obligation_by_id[item.obligation_id]
                returned_ids.add(item.obligation_id)
                output.append({
                    "article_number": obligation["article_number"],
                    "article_name": obligation["article_name"],
                    "chapter_number": obligation["chapter_number"],
                    "obligation_id": obligation["id"],
                    "severity": obligation.get("severity", "MEDIUM"),
                    "verdict": item.verdict,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "gap": item.gap,
                    "evidence": [
                        {"chunk_id": e["chunk_id"], "text": e["text"], "page": None, "similarity": e["similarity"]}
                        for e in obligation["evidence"]
                        if e["similarity"] >= MIN_SIMILARITY_FOR_EVIDENCE
                    ],
                })

            # If the LLM missed any obligation in the batch, retry the whole batch
            missing_ids = batch_ids - returned_ids
            if missing_ids:
                raise ValueError(f"LLM omitted {len(missing_ids)} obligation(s) from batch response")

            return output

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate limit" in err_str
            if attempt == MAX_RETRIES:
                print(f"  BATCH FAILED (final attempt), {len(batch)} obligations lost: {e}")
                return []
            wait = backoff if is_rate_limit else backoff / 2
            print(f"  retry {attempt}/{MAX_RETRIES} for batch of {len(batch)} "
                  f"({'rate limit' if is_rate_limit else 'error'}), waiting {wait:.1f}s")
            time.sleep(wait)
            backoff *= 2

    return []


def load_checkpoint():
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["obligation_id"] for r in existing}
        print(f"Resuming: {len(existing)} obligations already judged, will skip those.")
        return existing, done_ids
    except FileNotFoundError:
        return [], set()


def save_checkpoint(results):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    with open(EVIDENCE_INPUT_PATH, "r", encoding="utf-8") as f:
        all_obligations = json.load(f)
    results, done_ids = load_checkpoint()
    remaining = [o for o in all_obligations if o["id"] not in done_ids]
    print(f"{len(remaining)} obligations left to process (of {len(all_obligations)} total).")

    if not remaining:
        print("Nothing to do -- all obligations already judged.")
        return

    # ---- Step 0: applicability filter ----
    conditions_lookup = load_tagged_conditions()

    if conditions_lookup:
        for o in remaining:
            o["applicability_conditions"] = conditions_lookup.get(o["id"], [])
        company_profile = get_or_extract_company_profile()
        applicable, not_applicable = filter_applicable(remaining, company_profile)
    else:
        # tagged metadata missing -- fall back to old behavior, nothing skipped
        applicable, not_applicable = remaining, []

    print(f"Applicability filter: {len(not_applicable)} NOT_APPLICABLE (skipped entirely), "
          f"{len(applicable)} require evaluation.")

    for o in not_applicable:
        results.append(make_not_applicable(o))

    save_checkpoint(results)  # save the free applicability wins immediately

    remaining = applicable  # everything below only sees applicable obligations

    if not remaining:
        print(f"Done. {len(results)}/{len(all_obligations)} obligations judged -> {OUTPUT_PATH}")
        return

    # ---- Step 1: pre-filter ----

    # ---- Step 1: pre-filter ----
    needs_llm = []
    for obligation in remaining:
        if best_similarity(obligation) < PRE_FILTER_THRESHOLD:
            results.append(make_auto_not_met(obligation))
        else:
            needs_llm.append(obligation)

    print(f"Pre-filter: {len(remaining) - len(needs_llm)} auto-NOT_MET (skipped LLM), "
          f"{len(needs_llm)} need LLM judgment.")
    save_checkpoint(results)  # save the free pre-filtered wins immediately

    if not needs_llm:
        print(f"Done. {len(results)}/{len(all_obligations)} obligations judged -> {OUTPUT_PATH}")
        return

    # ---- Step 2: batch the rest ----
    batches = [needs_llm[i:i + BATCH_SIZE] for i in range(0, len(needs_llm), BATCH_SIZE)]
    print(f"Batched into {len(batches)} LLM calls (batch size {BATCH_SIZE}).")

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(JudgeVerdictBatchOutput)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

    save_every_n_batches = 3
    batches_since_save = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(judge_one_batch, batch, structured_llm, rate_limiter): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            batch_results = future.result()
            results.extend(batch_results)
            print(f"Batch done: +{len(batch_results)} results ({len(results)}/{len(all_obligations)} total)")

            batches_since_save += 1
            if batches_since_save >= save_every_n_batches:
                save_checkpoint(results)
                batches_since_save = 0

    save_checkpoint(results)
    print(f"Done. {len(results)}/{len(all_obligations)} obligations judged -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()