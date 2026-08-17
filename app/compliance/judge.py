# app/compliance/judge.py

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

from app.compliance.group_retriever import (
    ComplianceGroup,
    GroupEvidence,
)
from app.core.config import get_settings
from app.core.exceptions import (
    EndpointError,
    EvidenceValidationError,
    InvalidLLMResponseError,
    RateLimitError,
    TransientLLMError,
)
from app.core.llm_endpoint_pool import LLMEndpoint, LLMEndpointPool
from app.core.logger import get_logger
from app.models.sub_obligation import (
    EvidenceReference,
    SubObligationVerdict,
)

load_dotenv()

logger = get_logger()


@dataclass
class EvaluationMetrics:
    """Performance metrics tracked per group evaluation."""
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    validation_time: float = 0.0
    backoff_time: float = 0.0
    total_time: float = 0.0
    attempts: int = 0
    count_429: int = 0
    count_5xx: int = 0
    count_malformed_json: int = 0
    count_validation_failures: int = 0
    provider: str = ""
    model: str = ""
    endpoint_masked_key: str = ""


class VerdictList(list):
    """List subclass that permits attaching custom attributes like metrics."""
    metrics: EvaluationMetrics | None = None

    def __init__(self, items: list[SubObligationVerdict], metrics: EvaluationMetrics | None = None) -> None:
        super().__init__(items)
        self.metrics = metrics


class ComplianceJudge:
    """
    LLM-based judge for evaluating GDPR requirement groups.

    Responsibilities:
        1. Receive one GDPR requirement group.
        2. Receive retrieved company-policy evidence.
        3. Ask the LLM to evaluate each sub-obligation in a single request.
        4. Validate the returned JSON and strict evidence quotes.
        5. Convert the result into SubObligationVerdict objects.

    Multi-provider endpoint handling:
        A single ComplianceJudge instance is shared across all article/group worker threads.
        Each outbound HTTP call pulls the next endpoint from a thread-safe LLMEndpointPool
        and is protected by a global LLM concurrency semaphore.
    """

    BASE_BACKOFF_SECONDS = 2
    MAX_BACKOFF_SECONDS = 20

    _shared_llm_semaphore: threading.Semaphore | None = None
    _semaphore_lock = threading.Lock()

    def __init__(
        self,
        endpoint_pool: LLMEndpointPool | None = None,
        max_concurrent_requests: int | None = None,
    ) -> None:

        settings = get_settings()
        self.request_timeout = settings.LLM_REQUEST_TIMEOUT
        self.max_retries = settings.LLM_MAX_RETRIES

        self.endpoint_pool = (
            endpoint_pool
            or LLMEndpointPool.from_env()
        )

        max_concurrent = max_concurrent_requests or settings.MAX_CONCURRENT_LLM_REQUESTS
        with ComplianceJudge._semaphore_lock:
            if ComplianceJudge._shared_llm_semaphore is None:
                ComplianceJudge._shared_llm_semaphore = threading.Semaphore(max_concurrent)
        self.llm_semaphore = ComplianceJudge._shared_llm_semaphore

        logger.success(
            f"ComplianceJudge initialized with "
            f"{self.endpoint_pool.size} endpoint(s) across "
            f"provider(s): "
            f"{', '.join(sorted(self.endpoint_pool.providers()))} | "
            f"timeout={self.request_timeout}s | "
            f"max_retries={self.max_retries} | "
            f"global_max_concurrent_llm={max_concurrent}."
        )

    # ============================================================
    # PUBLIC
    # ============================================================

    def evaluate(
        self,
        group: ComplianceGroup,
        group_evidence: GroupEvidence,
    ) -> list[SubObligationVerdict]:
        """
        Evaluate every sub-obligation in one GDPR group.
        Priority chain: Gemini (1 attempt) -> OpenRouter (1 attempt) -> NVIDIA (1 attempt) -> Controlled fallback.
        """
        import os

        start_eval = time.perf_counter()
        metrics = EvaluationMetrics()

        prompt = self._build_prompt(
            group=group,
            group_evidence=group_evidence,
        )

        filter_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if filter_provider:
            provider_chain = [filter_provider]
        else:
            provider_chain = ["gemini", "openrouter", "nvidia"]

        last_error: Exception | None = None

        for idx, provider_name in enumerate(provider_chain):
            endpoint = self.endpoint_pool.next_endpoint_for_provider(provider_name)
            if not endpoint:
                continue

            logger.info(
                f"LLM_ATTEMPT | provider={provider_name} | "
                f"article={group.article_number} | group={group.group_id}"
            )
            metrics.attempts += 1

            t0 = time.perf_counter()
            try:
                raw_response = self._call_specific_endpoint(endpoint, prompt, metrics)
                dur = time.perf_counter() - t0
                metrics.llm_time += dur
                metrics.provider = endpoint.provider
                metrics.model = endpoint.model
                metrics.endpoint_masked_key = LLMEndpointPool.mask(endpoint.api_key)

                # Parse JSON and validate
                t_val = time.perf_counter()
                parsed = self._parse_json(raw_response)
                verdicts = self._validate_and_convert(
                    parsed=parsed,
                    group=group,
                    group_evidence=group_evidence,
                )
                metrics.validation_time += (time.perf_counter() - t_val)
                metrics.total_time = time.perf_counter() - start_eval

                logger.info(
                    f"LLM_SUCCESS | provider={provider_name} | "
                    f"article={group.article_number} | group={group.group_id} | "
                    f"latency={dur:.2f}s"
                )
                return VerdictList(verdicts, metrics=metrics)

            except RateLimitError as exc:
                last_error = exc
                if provider_name == "openrouter":
                    logger.warning(
                        f"OPENROUTER_RATE_LIMITED | Article={group.article_number} | "
                        f"Group={group.group_id}"
                    )
                logger.warning(
                    f"LLM_FAILURE | provider={provider_name} | "
                    f"article={group.article_number} | group={group.group_id} | "
                    f"error={exc}"
                )
            except Exception as exc:
                last_error = exc
                if provider_name == "nvidia":
                    logger.warning(
                        f"NVIDIA_FAILURE | Article={group.article_number} | "
                        f"Group={group.group_id} | error={exc}"
                    )
                logger.warning(
                    f"LLM_FAILURE | provider={provider_name} | "
                    f"article={group.article_number} | group={group.group_id} | "
                    f"error={exc}"
                )

            # Log fallback to next provider if available
            if idx + 1 < len(provider_chain):
                next_prov = provider_chain[idx + 1]
                logger.info(
                    f"LLM_FALLBACK | from={provider_name} | to={next_prov}"
                )

        # All providers in chain failed or unavailable
        metrics.total_time = time.perf_counter() - start_eval
        logger.warning(
            f"LLM_FALLBACK_EXHAUSTED | article={group.article_number} | "
            f"group={group.group_id} | last_error={last_error}"
        )

        fallback_reason = (
            f"Unable to obtain a valid assessment from LLM providers: {last_error or 'No endpoints available'}"
        )
        fallback_verdicts = [
            SubObligationVerdict(
                obligation_id=ob.id,
                status="INSUFFICIENT_EVIDENCE",
                reason=fallback_reason,
                evidence=(),
                confidence=0.0,
            )
            for ob in group.obligations
        ]
        return VerdictList(fallback_verdicts, metrics=metrics)

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        group: ComplianceGroup,
        group_evidence: GroupEvidence,
    ) -> str:

        obligations = []
        for obligation in group.obligations:
            obligations.append(
                {
                    "id": obligation.id,
                    "legal_text": obligation.legal_text,
                    "plain_summary": obligation.plain_summary,
                    "evidence_prompt": obligation.evidence_prompt,
                    "applicability_condition": obligation.applicability_condition,
                }
            )

        evidence = []
        for item in group_evidence.evidence:
            evidence.append(
                {
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                    "text": item.text,
                }
            )

        return f"""You are a strict GDPR compliance evidence judge evaluating ONE GDPR requirement group.
Your decision MUST be based ONLY on the company-policy evidence supplied below.
Do NOT use outside information or assume unstated company behavior.

============================================================
GDPR REQUIREMENT GROUP
============================================================
Article: {group.article_number}
Group ID: {group.group_id}
Principle: {group.principle}
Condition Logic: {group.condition_logic}
Requirement Summary: {group.requirement_summary}
Applicability Condition: {group.applicability_condition}

============================================================
SUB-OBLIGATIONS TO EVALUATE
============================================================
{json.dumps(obligations, indent=2, ensure_ascii=False)}

============================================================
COMPANY POLICY EVIDENCE
============================================================
{json.dumps(evidence, indent=2, ensure_ascii=False)}

============================================================
STATUS DEFINITIONS
============================================================
MET: Evidence clearly demonstrates the obligation is satisfied.
PARTIALLY_MET: Evidence demonstrates some aspects, but not complete obligation.
NOT_MET: Evidence indicates the obligation is NOT satisfied.
CONFLICTING: Evidence conflicts with requirement.
INSUFFICIENT_EVIDENCE: Evidence is insufficient to evaluate.
NOT_APPLICABLE: Obligation does not apply based on applicability conditions.

============================================================
IMPORTANT RULES
============================================================
1. Evaluate EVERY supplied sub-obligation ID. Do NOT skip any.
2. Do NOT invent new obligation IDs or duplicates.
3. Evidence references MUST use existing chunk_id.
4. Quotes must be EXACT short text quotes from the supplied evidence. Do NOT invent quotes.
5. Confidence must be between 0.0 and 1.0.
6. Return JSON only. No markdown fences.

============================================================
OUTPUT FORMAT
============================================================
{{
  "sub_obligations": [
    {{
      "id": "{group.obligations[0].id if group.obligations else 'id'}",
      "status": "MET",
      "confidence": 0.95,
      "reason": "Short evidence-grounded explanation.",
      "evidence": [
        {{
          "chunk_id": "chunk_id_from_above",
          "quote": "Exact quote from policy text."
        }}
      ]
    }}
  ]
}}
"""

    # ============================================================
    # LLM CALL (Guarded by global semaphore)
    # ============================================================

    def _call_specific_endpoint(
        self,
        endpoint: LLMEndpoint,
        prompt: str,
        metrics: EvaluationMetrics,
    ) -> str:
        """
        Execute ONE attempt against a specified LLMEndpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint.api_key}",
        }

        payload = {
            "model": endpoint.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert GDPR compliance auditor. "
                        "Evaluate compliance against retrieved policy evidence strict accuracy. "
                        "Return strictly valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        with self.llm_semaphore:
            try:
                response = requests.post(
                    endpoint.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
            except requests.exceptions.Timeout as exc:
                metrics.count_5xx += 1
                self.endpoint_pool.record_failure(endpoint, is_5xx=True)
                raise TransientLLMError(
                    f"Timeout calling {endpoint.provider} ({LLMEndpointPool.mask(endpoint.api_key)}): {exc}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                self.endpoint_pool.record_failure(endpoint)
                raise TransientLLMError(
                    f"Network error calling {endpoint.provider} ({LLMEndpointPool.mask(endpoint.api_key)}): {exc}"
                ) from exc

        if response.status_code == 429:
            metrics.count_429 += 1
            self.endpoint_pool.mark_cooldown(endpoint, duration_seconds=30.0)
            raise RateLimitError(
                f"Rate limited (429) on {endpoint.provider} key {LLMEndpointPool.mask(endpoint.api_key)}."
            )

        if response.status_code >= 500:
            metrics.count_5xx += 1
            self.endpoint_pool.record_failure(endpoint, is_5xx=True)
            raise TransientLLMError(
                f"Server error ({response.status_code}) on {endpoint.provider} key {LLMEndpointPool.mask(endpoint.api_key)}."
            )

        if response.status_code != 200:
            raise EndpointError(
                f"HTTP status {response.status_code} from {endpoint.provider}: {response.text[:200]}"
            )

        self.endpoint_pool.record_success(endpoint)

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise InvalidLLMResponseError(
                    f"Empty LLM response content from {endpoint.provider}."
                )
            return content
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            metrics.count_malformed_json += 1
            raise InvalidLLMResponseError(
                f"Failed to parse LLM response JSON from {endpoint.provider}: {exc}"
            ) from exc

    @classmethod
    def _backoff_seconds(
        cls,
        attempt: int,
    ) -> float:
        return min(
            cls.BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            cls.MAX_BACKOFF_SECONDS,
        )

    # ============================================================
    # JSON PARSER (Robust with outer object extraction)
    # ============================================================

    @staticmethod
    def _parse_json(
        raw_response: str,
    ) -> dict[str, Any]:

        text = raw_response.strip()

        # 1. Remove markdown fences
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # 2. Try direct parsing
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 3. Try safe outer JSON object extraction using regex
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise InvalidLLMResponseError(
            f"Could not extract valid JSON object from response: '{raw_response[:150]}...'"
        )

    # ============================================================
    # VALIDATION (Strict Obligation + Soft Evidence Quote Check)
    # ============================================================

    def _validate_and_convert(
        self,
        parsed: dict[str, Any],
        group: ComplianceGroup,
        group_evidence: GroupEvidence,
    ) -> list[SubObligationVerdict]:

        raw_results = parsed.get("sub_obligations")

        if not isinstance(raw_results, list):
            raise InvalidLLMResponseError(
                "Judge response is missing 'sub_obligations' list."
            )

        expected_ids = [obligation.id for obligation in group.obligations]
        expected_ids_set = set(expected_ids)

        returned_items = [item for item in raw_results if isinstance(item, dict)]
        returned_ids = [item.get("id") for item in returned_items]
        returned_ids_set = set(returned_ids)

        # 1. Reject duplicate obligation IDs
        if len(returned_ids) != len(returned_ids_set):
            raise InvalidLLMResponseError(
                f"Judge returned duplicate obligation IDs in response: {returned_ids}"
            )

        # 2. Reject missing or unknown obligation IDs
        missing = expected_ids_set - returned_ids_set
        extra = returned_ids_set - expected_ids_set

        if missing:
            raise InvalidLLMResponseError(
                f"Judge omitted expected sub-obligations: {sorted(missing)}"
            )

        if extra:
            raise InvalidLLMResponseError(
                f"Judge returned unknown sub-obligations: {sorted(extra)}"
            )

        valid_statuses = {
            "MET",
            "PARTIALLY_MET",
            "NOT_MET",
            "CONFLICTING",
            "INSUFFICIENT_EVIDENCE",
            "NOT_APPLICABLE",
        }

        # Build map of chunk_id -> evidence text
        evidence_map = {
            evidence.chunk_id: evidence.text
            for evidence in group_evidence.evidence
        }
        first_chunk_id = group_evidence.evidence[0].chunk_id if group_evidence.evidence else None

        results = []

        for item in returned_items:

            obligation_id = str(item.get("id", ""))
            status = str(item.get("status", "")).upper()
            raw_confidence = item.get("confidence")
            reason = item.get("reason")
            evidence_items = item.get("evidence", [])

            if status not in valid_statuses:
                raise InvalidLLMResponseError(
                    f"Invalid status for {obligation_id}: '{status}'"
                )

            # Safe confidence normalization
            try:
                conf_val = float(raw_confidence)
                if math.isnan(conf_val) or math.isinf(conf_val):
                    conf_val = 0.0
                confidence = max(0.0, min(1.0, conf_val))
            except (TypeError, ValueError):
                confidence = 0.0

            if not isinstance(reason, str) or not reason.strip():
                reason = "Evaluated requirement based on policy evidence."

            if not isinstance(evidence_items, list):
                evidence_items = []

            references = []

            for evidence_item in evidence_items:

                if not isinstance(evidence_item, dict):
                    continue

                chunk_id = str(evidence_item.get("chunk_id", ""))
                quote = str(evidence_item.get("quote", ""))

                # Fallback matching for unknown chunk_id
                if chunk_id not in evidence_map:
                    # Attempt to find chunk containing quote
                    found_chunk = None
                    if quote.strip():
                        for cid, ctext in evidence_map.items():
                            if self._quote_exists_in_text(quote, ctext):
                                found_chunk = cid
                                break
                    if found_chunk:
                        chunk_id = found_chunk
                    elif first_chunk_id:
                        chunk_id = first_chunk_id
                    else:
                        continue  # Skip invalid chunk if no evidence available

                if not quote.strip():
                    if chunk_id in evidence_map:
                        quote = evidence_map[chunk_id][:150] + "..."
                    else:
                        continue

                # Soft Quote Validation: check if quote exists, or fall back to snippet
                chunk_text = evidence_map.get(chunk_id, "")
                if chunk_text and not self._quote_exists_in_text(quote, chunk_text):
                    logger.warning(
                        f"Quote for {obligation_id} not strictly matched in chunk '{chunk_id}'. "
                        f"Applying soft quote match."
                    )
                    # Extract a matching snippet or use clean quote snippet
                    words = quote.split()
                    if len(words) > 3:
                        # Try prefix words match
                        short_quote = " ".join(words[:5])
                        if short_quote.lower() in chunk_text.lower():
                            quote = short_quote
                        else:
                            quote = chunk_text[:150] + "..."
                    else:
                        quote = chunk_text[:150] + "..."

                references.append(
                    EvidenceReference(
                        chunk_id=chunk_id,
                        quote=quote,
                    )
                )

            results.append(
                SubObligationVerdict(
                    obligation_id=obligation_id,
                    status=status,
                    reason=reason,
                    evidence=tuple(references),
                    confidence=confidence,
                )
            )

        return results

    @staticmethod
    def _quote_exists_in_text(quote: str, text: str) -> bool:
        """
        Whitespace-normalized substring matching to check if quote exists in retrieved text.
        Supports ellipsis (...), punctuation stripping, and sub-phrase matching.
        """
        if not quote or not text:
            return False

        norm_quote = re.sub(r"\s+", " ", quote).strip().lower()
        norm_text = re.sub(r"\s+", " ", text).strip().lower()

        if norm_quote in norm_text:
            return True

        # Strip punctuation
        clean_quote_str = re.sub(r"[^\w\s]", "", norm_quote).strip()
        clean_text_str = re.sub(r"[^\w\s]", "", norm_text).strip()
        if clean_quote_str and clean_quote_str in clean_text_str:
            return True

        # Handle ellipsis inside or at the end of quotes (e.g. "foo...bar" or "foo...")
        fragments = [f.strip() for f in re.split(r"\.{3,}|\u2026", norm_quote) if len(f.strip()) >= 4]
        if fragments and any(f in norm_text for f in fragments):
            return True

        clean_quote = norm_quote.strip(".… \t\n")
        if len(clean_quote) >= 4 and clean_quote in norm_text:
            return True

        # Handle word prefix & suffix match for long quotes truncated with trailing/middle dots
        words = clean_quote.split()
        if len(words) >= 4:
            prefix = " ".join(words[:3])
            suffix = " ".join(words[-3:])
            if prefix in norm_text or suffix in norm_text:
                return True

        return False