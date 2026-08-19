"""One-request NVIDIA NIM connectivity diagnostic; it never runs the RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Support direct execution (``python tests/test_nvidia_endpoint.py``) without
# depending on the full application or its RAG integrations.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm_endpoint_pool import LLMEndpointPool


def main() -> int:
    load_dotenv()
    print("=" * 60)
    print("NVIDIA ENDPOINT TEST")
    print("=" * 60)

    try:
        pool = LLMEndpointPool.from_env()
    except ValueError as exc:
        print(f"CONFIGURATION ERROR: {exc}")
        return 1

    endpoints = []
    for _ in range(pool.size):
        endpoint = pool.next_endpoint()
        if endpoint.provider == "nvidia":
            endpoints.append(endpoint)

    if not endpoints:
        print("CONFIGURATION ERROR: No NVIDIA NIM endpoint is configured.")
        return 1

    endpoint = endpoints[0]
    print(f"Provider: {endpoint.provider}")
    print(f"Model: {endpoint.model}")
    print(f"Endpoint count: {len(endpoints)}")
    for index, configured in enumerate(endpoints, start=1):
        print(
            f"Endpoint {index}: {configured.provider} "
            f"{LLMEndpointPool.mask(configured.api_key)}"
        )

    print("\nSending one test request...")
    try:
        response = requests.post(
            endpoint.base_url,
            headers={
                "Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": endpoint.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly this JSON and nothing else:\n"
                            '{"status":"OK"}'
                        ),
                    }
                ],
            },
            timeout=30,
        )
        print(f"HTTP status: {response.status_code}")
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        print("\nResponse:")
        print(content)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\nNVIDIA ENDPOINT TEST: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
