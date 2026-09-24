import os
import httpx
import time
from app.logger import logger

from app.providers.base import AIProvider

from app.exceptions import (
    AIConnectionError,
    AIResponseError,
    AIServiceError,
    AITimeoutError,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Retry decorator applied to the entire OllamaProvider class.
# If generate() or generate_structured() raises a connection or timeout error,
# tenacity will retry up to 3 times with exponential backoff (1s → 2s → 4s)
# before giving up. Non-retryable errors (e.g. 4xx/5xx HTTP responses) fail fast.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=4,
    ),
    retry=retry_if_exception_type(
        (
            AIConnectionError,
            AITimeoutError,
        )
    ),
    reraise=True,
)
class OllamaProvider(AIProvider):
    def __init__(self):
        # Read the Ollama base URL from environment; fall back to the default
        # local-instance address so the service runs out-of-the-box on Docker.
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        # Model identifier — must match a model pulled into the local Ollama instance.
        self.model = os.getenv(
            "OLLAMA_MODEL",
            "qwen3:1.7b",
        )

        # Inference timeout in seconds. Cast to float so sub-second precision is supported.
        self.timeout = float(
            os.getenv(
                "AI_TIMEOUT",
                "120",
            )
        )

    async def generate_structured(self, prompt: str, schema: dict) -> dict:
        """Send a prompt to Ollama with structured JSON output according to the provided schema.

        The Ollama API accepts a `format` field containing a JSON schema; the model
        is constrained to return only objects matching that schema. We capture the
        raw response dict so the caller can validate it against the Pydantic model.

        Args:
            prompt: Text instruction for the AI model.
            schema: JSON schema definition (from Pydantic model_json_schema()).

        Returns:
            Dictionary with at least a "response" key containing the model's JSON string.
        """
        start_time = time.perf_counter()

        try:
            # Context-manager ensures the HTTP client is closed even if an exception occurs.
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,          # non-streaming — wait for the full response
                        "format": schema,         # constrains output to the requested JSON shape
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                latency = time.perf_counter() - start_time
                logger.info(
                    "AI inference completed | model=%s | latency=%.2fs",
                    self.model,
                    latency,
                )

                data = response.json()

                # Log token counts for observability (useful for cost/throughput tracking).
                input_tokens = data.get(
                    "prompt_eval_count",
                    0,
                )

                output_tokens = data.get(
                    "eval_count",
                    0,
                )

                logger.info(
                    (
                        "AI inference completed | "
                        "model=%s | "
                        "latency=%.2fs | "
                        "input_tokens=%d | "
                        "output_tokens=%d"
                    ),
                    self.model,
                    latency,
                    input_tokens,
                    output_tokens,
                )

                # Ollama's /api/generate may omit "response" on malformed output.
                if "response" not in data:
                    raise AIResponseError("Ollama returned an invalid response.")

                return data
        except httpx.ConnectError as exc:
            # Network unreachable / DNS failure / connection refused.
            raise AIConnectionError("Cannot connect to Ollama.") from exc

        except httpx.TimeoutException as exc:
            # Request exceeded the configured AI_TIMEOUT.
            raise AITimeoutError("Ollama inference timed out.") from exc

        except httpx.HTTPStatusError as exc:
            # 4xx/5xx from Ollama — forward as a generic response error.
            raise AIResponseError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc

    async def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and retrieve the raw text response.

        Used for unstructured tasks (summarize, rewrite, translate) where the
        caller only needs the generated string, not typed JSON.

        Args:
            prompt: Text instruction for the AI model.

        Returns:
            Generated text from the model.
        """
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=self.timeout,
                )
                response.raise_for_status()

                latency = time.perf_counter() - start_time
                logger.info(
                    "AI inference completed | model=%s | latency=%.2fs",
                    self.model,
                    latency,
                )

                data = response.json()

                # Log token counts for observability.
                input_tokens = data.get(
                    "prompt_eval_count",
                    0,
                )

                output_tokens = data.get(
                    "eval_count",
                    0,
                )

                logger.info(
                    (
                        "AI inference completed | "
                        "model=%s | "
                        "latency=%.2fs | "
                        "input_tokens=%d | "
                        "output_tokens=%d"
                    ),
                    self.model,
                    latency,
                    input_tokens,
                    output_tokens,
                )

                if "response" not in data:
                    raise AIResponseError("Ollama returned an invalid response.")

                # Extract only the text payload; the wrapper dict is not needed here.
                return data["response"]
        except httpx.ConnectError as exc:
            raise AIConnectionError("Cannot connect to Ollama.") from exc

        except httpx.TimeoutException as exc:
            raise AITimeoutError("Ollama inference timed out.") from exc

        except httpx.HTTPStatusError as exc:
            raise AIResponseError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc
