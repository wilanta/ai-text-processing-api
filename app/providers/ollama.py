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

        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        self.model = os.getenv(
            "self.model",
            "qwen3:1.7b",
        )

        self.timeout = float(
            os.getenv(
                "AI_TIMEOUT",
                "120",
            )
        )

    async def generate_structured(self, prompt: str, schema: dict) -> dict:
        """Send a prompt to Ollama with structured JSON format according to the schema.

        Args:
            prompt: Text instruction for the AI model.
            schema: JSON schema definition (from Pydantic model_json_schema()).

        Returns:
            Dictionary containing the JSON response from Ollama.
        """
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient() as client:
                response = await httpx.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": schema,
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

                return data
        except httpx.ConnectError as exc:
            raise AIConnectionError("Cannot connect to Ollama.") from exc

        except httpx.TimeoutException as exc:
            raise AITimeoutError("Ollama inference timed out.") from exc

        except httpx.HTTPStatusError as exc:
            raise AIResponseError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc

    async def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and retrieve the raw text response.

        Args:
            prompt: Text instruction for the AI model.

        Returns:
            Generated text from the model.
        """
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient() as client:
                response = await httpx.post(
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

                return data["response"]
        except httpx.ConnectError as exc:
            raise AIConnectionError("Cannot connect to Ollama.") from exc

        except httpx.TimeoutException as exc:
            raise AITimeoutError("Ollama inference timed out.") from exc

        except httpx.HTTPStatusError as exc:
            raise AIResponseError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc
