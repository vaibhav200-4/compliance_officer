"""
applicability/tag_obligations.py
------------------------------------------
Run this ONCE (or whenever gdpr_articles.json changes) to tag every GDPR
obligation with applicability_conditions -- i.e. which company-specific
conditions must be true for that obligation to actually apply.

Rate-limited + retried + checkpointed, so if you hit a quota error partway
through, just re-run the script -- it skips obligations already tagged.

Input:  dataa/article_metadata1.json
Output: dataa/article_metadata1_tagged.json
"""

import json
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from typing import List

from applicability.conditions import APPLICABILITY_CONDITIONS
#from report.llm_called.llm_client import get_llm   # same client used by judge_obligations.py
from report.llm_called.llm_client import get_gemini_llm
# ---------- CONFIG ----------
INPUT_PATH = "dataa/article_metadata1.json"
OUTPUT_PATH = "dataa/article_metadata_tagged.json"

MAX_WORKERS = 3
MAX_REQUESTS_PER_MINUTE = 12     # keep safely under whatever provider limit you're on
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 3
# -----------------------------


class ObligationTag(BaseModel):
    id: str
    applicability_conditions: List[str]
    reasoning: str


TAGGING_PROMPT = """You are tagging a GDPR obligation with applicability conditions.

Return which conditions MUST be true for a company for this obligation to apply.
If the obligation applies to ALL data controllers/processors universally
(e.g. core principles, general rights, general security), return an empty list.

Valid conditions (use ONLY these exact strings, nothing else):
{conditions}

Obligation ID: {id}
Title: {title}
Text: {text}

Return ONLY JSON: {{"id": "{id}", "applicability_conditions": [...], "reasoning": "<one line>"}}
"""


class RateLimiter:
    """Shared token-bucket limiter -- same pattern as judge_obligations.py."""
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


def tag_single_obligation(obligation, structured_llm, rate_limiter):
    prompt = TAGGING_PROMPT.format(
        conditions=", ".join(APPLICABILITY_CONDITIONS),
        id=obligation["id"],
        title=obligation.get("title", obligation.get("article_name", "")),
        text=obligation.get("text", obligation.get("description", ""))[:1200],
    )

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        rate_limiter.wait_for_slot()
        try:
            result: ObligationTag = structured_llm.invoke(prompt)
            return result.model_dump()
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str
            if attempt == MAX_RETRIES:
                raise
            wait = backoff if is_rate_limit else backoff / 2
            print(f"  retry {attempt}/{MAX_RETRIES} for {obligation['id']} "
                  f"({'rate limit' if is_rate_limit else 'error'}), waiting {wait:.1f}s")
            time.sleep(wait)
            backoff *= 2


def load_checkpoint():
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {o["id"] for o in existing if "applicability_conditions" in o}
        print(f"Resuming: {len(done_ids)} obligations already tagged, will skip those.")
        return existing, done_ids
    except FileNotFoundError:
        return [], set()


def save_checkpoint(all_metadata):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)


def tag_all():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original_metadata = json.load(f)

    tagged_so_far, done_ids = load_checkpoint()

    # Build working list: use checkpoint version where available, else original
    tagged_by_id = {o["id"]: o for o in tagged_so_far}
    working_metadata = []
    for obl in original_metadata:
        if obl["id"] in tagged_by_id:
            working_metadata.append(tagged_by_id[obl["id"]])
        else:
            working_metadata.append(obl)

    remaining = [o for o in working_metadata if o["id"] not in done_ids]
    print(f"{len(remaining)} obligations left to tag (of {len(working_metadata)} total).")

    if not remaining:
        print(f"Nothing to do. Already complete -> {OUTPUT_PATH}")
        return

    llm = get_gemini_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ObligationTag)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

    obligation_by_id = {o["id"]: o for o in working_metadata}
    save_every_n = 10
    processed_since_save = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(tag_single_obligation, obl, structured_llm, rate_limiter): obl["id"]
            for obl in remaining
        }

        for future in as_completed(futures):
            obl_id = futures[future]
            try:
                tag_result = future.result()
                obligation_by_id[obl_id]["applicability_conditions"] = tag_result.get("applicability_conditions", [])
                obligation_by_id[obl_id]["_tag_reasoning"] = tag_result.get("reasoning", "")
                done_ids.add(obl_id)
            except Exception as e:
                print(f"  FAILED (all retries exhausted) for {obl_id}: {e} -- left untagged, will retry on next run")

            processed_since_save += 1
            if processed_since_save >= save_every_n:
                save_checkpoint(list(obligation_by_id.values()))
                processed_since_save = 0
                print(f"  checkpoint saved: {len(done_ids)}/{len(working_metadata)} tagged")

    save_checkpoint(list(obligation_by_id.values()))
    still_untagged = [o["id"] for o in working_metadata if o["id"] not in done_ids]
    print(f"\nDone. {len(done_ids)}/{len(working_metadata)} tagged -> {OUTPUT_PATH}")
    if still_untagged:
        print(f"{len(still_untagged)} obligations still failed -- re-run this script to retry just those.")


if __name__ == "__main__":
    tag_all()