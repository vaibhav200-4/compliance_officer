"""Local tests for LLMEndpointPool; no external services are contacted."""

from app.core.llm_endpoint_pool import LLMEndpoint, LLMEndpointPool


def test_round_robin_deduplicates_blank_and_duplicate_keys() -> None:
    pool = LLMEndpointPool(
        [
            LLMEndpoint("nvidia", "https://example.test", "key1", "model"),
            LLMEndpoint("nvidia", "https://example.test", " key2 ", "model"),
            LLMEndpoint("nvidia", "https://example.test", "", "model"),
            LLMEndpoint("nvidia", "https://example.test", "key1", "model"),
            LLMEndpoint("nvidia", "https://example.test", "key3", "model"),
        ]
    )

    assert pool.size == 3
    assert [pool.next_endpoint().api_key for _ in range(6)] == [
        "key1",
        "key2",
        "key3",
        "key1",
        "key2",
        "key3",
    ]
    assert LLMEndpointPool.mask("abcdefgh") == "...efgh"
    assert LLMEndpointPool.mask("abcd") == "****"
