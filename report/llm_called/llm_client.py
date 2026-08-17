import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Ordered fallback chain -- tried in sequence when the previous model's
# quota/rate-limit is hit. Each Groq model has its OWN separate daily quota,
# so chaining models (not providers) multiplies your effective daily budget
# without needing a second API key.
MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "groq/compound-mini",
    "qwen/qwen3.6-27b",
]


def _is_quota_error(e: Exception) -> bool:
    err = str(e).lower()
    return (
        "429" in err
        or "rate limit" in err
        or "rate_limit" in err
        or "resource_exhausted" in err
        or "quota" in err
        or "too many requests" in err
    )


class _FallbackLLM:
    """
    Drop-in replacement for a single ChatGroq instance that transparently
    walks MODEL_CHAIN on quota/rate-limit errors. Supports the same interface
    every caller already uses: llm.with_structured_output(Schema).invoke(prompt)
    or llm.invoke(prompt) directly.
    """

    def __init__(self, models, temperature, api_key):
        self.models = models
        self.temperature = temperature
        self.api_key = api_key
        self._schema = None

    def with_structured_output(self, schema):
        self._schema = schema
        return self

    def _build(self, model_name):
        base = ChatGroq(
            model=model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            timeout=60,
            max_retries=1,
        )
        return base.with_structured_output(self._schema) if self._schema else base

    def invoke(self, prompt):
        last_err = None
        for i, model_name in enumerate(self.models):
            try:
                client = self._build(model_name)
                result = client.invoke(prompt)
                if i > 0:
                    print(f"  [fallback] succeeded using model: {model_name}")
                return result
            except Exception as e:
                if _is_quota_error(e):
                    print(f"  [fallback] {model_name} quota/rate-limited, trying next model in chain...")
                    last_err = e
                    time.sleep(1)  # tiny pause before switching models
                    continue
                else:
                    # non-quota error (bad prompt, schema mismatch, etc.) -- don't
                    # burn through the whole chain for something retrying won't fix
                    raise
        # every model in the chain was exhausted
        raise RuntimeError(
            f"All models in MODEL_CHAIN exhausted their quota. Last error: {last_err}"
        )


def get_gemini_llm(temperature: float = 0, model: str = "gemini-3.1-flash-lite"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key, timeout=60, max_retries=1)


def get_llm(temperature: float = 0, model: str | None = None):
    """
    If `model` is explicitly passed, behaves exactly as before -- a single
    fixed ChatGroq instance, no fallback (useful when you deliberately want
    one specific model, e.g. the cheap 8b-instant for a simple task).

    If `model` is None (default), returns a fallback-chain wrapper that walks
    through MODEL_CHAIN automatically whenever a quota/rate-limit error hits.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    if model:
        return ChatGroq(model=model, temperature=temperature, api_key=api_key, timeout=60, max_retries=1)

    return _FallbackLLM(models=MODEL_CHAIN, temperature=temperature, api_key=api_key)