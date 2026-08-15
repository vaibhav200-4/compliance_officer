# app/compliance/judge.py

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from app.compliance.group_retriever import (
    ComplianceGroup,
    GroupEvidence,
)
from app.models.sub_obligation import (
    EvidenceReference,
    SubObligationVerdict,
)

load_dotenv()


class ComplianceJudge:
    """
    LLM-based judge for evaluating GDPR requirement groups.

    Responsibilities:
        1. Receive one GDPR requirement group.
        2. Receive retrieved company-policy evidence.
        3. Ask the LLM to evaluate each sub-obligation.
        4. Validate the returned JSON.
        5. Convert the result into SubObligationVerdict objects.

    It does NOT:
        - retrieve from Pinecone
        - build embeddings
        - aggregate ALL/ANY/SINGLE
        - generate the final compliance table
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
        )

        self.model = (
            model
            or os.getenv("OPENROUTER_MODEL")
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        if not self.model:
            raise ValueError(
                "OPENROUTER_MODEL is not configured."
            )

        self.url = (
            "https://openrouter.ai/api/v1/chat/completions"
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
        """

        prompt = self._build_prompt(
            group=group,
            group_evidence=group_evidence,
        )

        raw_response = self._call_llm(prompt)

        parsed = self._parse_json(
            raw_response
        )

        return self._validate_and_convert(
            parsed=parsed,
            group=group,
            group_evidence=group_evidence,
        )

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
                    "applicability_condition": (
                        obligation.applicability_condition
                    ),
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

        return f"""
You are a strict GDPR compliance evidence judge.

You are evaluating ONE GDPR requirement group.

Your decision MUST be based ONLY on the company-policy
evidence supplied below.

Do NOT use outside information.
Do NOT assume that the company does something unless the
provided policy explicitly supports it.

============================================================
GDPR REQUIREMENT GROUP
============================================================

Article:
{group.article_number}

Group ID:
{group.group_id}

Principle:
{group.principle}

Condition Logic:
{group.condition_logic}

Requirement Summary:
{group.requirement_summary}

Applicability Condition:
{group.applicability_condition}

Assessment Rules:
{json.dumps(
    group.assessment_rules,
    indent=2,
    ensure_ascii=False,
)}

============================================================
SUB-OBLIGATIONS
============================================================

{json.dumps(
    obligations,
    indent=2,
    ensure_ascii=False,
)}

============================================================
COMPANY POLICY EVIDENCE
============================================================

{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False,
)}

============================================================
STATUS DEFINITIONS
============================================================

MET:
The evidence clearly demonstrates that the obligation
is satisfied.

PARTIALLY_MET:
The evidence demonstrates that some aspects are satisfied,
but the complete obligation is not demonstrated.

NOT_MET:
The available evidence indicates that the obligation is
not satisfied.

CONFLICTING:
The policy contains evidence that conflicts with the
requirement.

INSUFFICIENT_EVIDENCE:
The supplied evidence does not provide enough information
to determine compliance.

NOT_APPLICABLE:
The obligation does not apply based on the supplied
applicability information.

============================================================
IMPORTANT RULES
============================================================

1. Evaluate EVERY supplied sub-obligation.

2. Do NOT skip an obligation.

3. Use ONLY the supplied company-policy evidence.

4. Every evidence reference MUST use an existing chunk_id.

5. Quotes must be exact text from the supplied evidence.

6. Do not invent quotes.

7. Keep quotes short and directly relevant.

8. Confidence must be between 0.0 and 1.0.

9. Do not determine the final group status.
   Evaluate individual sub-obligations only.

10. Return JSON only.

============================================================
OUTPUT
============================================================

Return exactly:

{{
  "sub_obligations": [
    {{
      "id": "5.1.f.1",
      "status": "MET",
      "confidence": 0.95,
      "reason": "Short evidence-based explanation.",
      "evidence": [
        {{
          "chunk_id": "sample_company_sec7_chunk0",
          "quote": "Exact quote from policy."
        }}
      ]
    }}
  ]
}}

No Markdown.
No ```json.
No additional text.
JSON only.
"""

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
    ) -> str:

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict GDPR compliance "
                        "evidence judge. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:

            raise ValueError(
                "Unexpected OpenRouter response format."
            ) from exc

    # ============================================================
    # JSON PARSER
    # ============================================================

    @staticmethod
    def _parse_json(
        raw_response: str,
    ) -> dict[str, Any]:

        text = raw_response.strip()

        # Remove accidental Markdown fences.
        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError as exc:

            print(
                "\n========== INVALID JUDGE JSON =========="
            )
            print(text)
            print(
                "========================================\n"
            )

            raise ValueError(
                f"Judge returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Judge response must be a JSON object."
            )

        return parsed

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_and_convert(
        self,
        parsed: dict[str, Any],
        group: ComplianceGroup,
        group_evidence: GroupEvidence,
    ) -> list[SubObligationVerdict]:

        raw_results = parsed.get(
            "sub_obligations"
        )

        if not isinstance(
            raw_results,
            list,
        ):
            raise ValueError(
                "Judge response is missing "
                "'sub_obligations' list."
            )

        expected_ids = {
            obligation.id
            for obligation in group.obligations
        }

        returned_ids = {
            item.get("id")
            for item in raw_results
            if isinstance(item, dict)
        }

        missing = (
            expected_ids - returned_ids
        )

        extra = (
            returned_ids - expected_ids
        )

        if missing:
            raise ValueError(
                f"Judge omitted sub-obligations: "
                f"{sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Judge returned unknown "
                f"sub-obligations: {sorted(extra)}"
            )

        valid_statuses = {
            "MET",
            "PARTIALLY_MET",
            "NOT_MET",
            "CONFLICTING",
            "INSUFFICIENT_EVIDENCE",
            "NOT_APPLICABLE",
        }

        valid_chunk_ids = {
            evidence.chunk_id
            for evidence in group_evidence.evidence
        }

        results = []

        for item in raw_results:

            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "Each sub-obligation result "
                    "must be an object."
                )

            obligation_id = item.get("id")

            status = item.get(
                "status"
            )

            confidence = item.get(
                "confidence"
            )

            reason = item.get(
                "reason"
            )

            evidence_items = item.get(
                "evidence",
                [],
            )

            if status not in valid_statuses:
                raise ValueError(
                    f"Invalid status for "
                    f"{obligation_id}: {status}"
                )

            if not isinstance(
                confidence,
                (int, float),
            ):
                raise ValueError(
                    f"Invalid confidence for "
                    f"{obligation_id}."
                )

            confidence = float(
                confidence
            )

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Confidence for "
                    f"{obligation_id} must be "
                    f"between 0 and 1."
                )

            if not isinstance(
                reason,
                str,
            ):
                raise ValueError(
                    f"Missing reason for "
                    f"{obligation_id}."
                )

            if not isinstance(
                evidence_items,
                list,
            ):
                raise ValueError(
                    f"Evidence for "
                    f"{obligation_id} must be a list."
                )

            references = []

            for evidence_item in evidence_items:

                if not isinstance(
                    evidence_item,
                    dict,
                ):
                    raise ValueError(
                        "Evidence reference must "
                        "be an object."
                    )

                chunk_id = evidence_item.get(
                    "chunk_id"
                )

                quote = evidence_item.get(
                    "quote"
                )

                if chunk_id not in valid_chunk_ids:
                    raise ValueError(
                        f"Judge referenced unknown "
                        f"chunk_id '{chunk_id}' "
                        f"for {obligation_id}."
                    )

                if not isinstance(
                    quote,
                    str,
                ) or not quote.strip():
                    raise ValueError(
                        f"Empty quote for "
                        f"{obligation_id}."
                    )

                # Ensure the quote actually exists
                # in the retrieved policy chunk.
                matching_evidence = next(
                    (
                        evidence
                        for evidence
                        in group_evidence.evidence
                        if evidence.chunk_id
                        == chunk_id
                    ),
                    None,
                )

                if matching_evidence is None:
                    raise ValueError(
                        f"Evidence chunk "
                        f"{chunk_id} not found."
                    )

                
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
                    evidence=tuple(
                        references
                    ),
                    confidence=confidence,
                )
            )

        return results