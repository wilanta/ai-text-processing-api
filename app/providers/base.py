from abc import ABC, abstractmethod


# Abstract base class that every AI provider must implement.
# Adding a new provider (e.g. OpenAI, Anthropic) only requires subclassing
# this and implementing the two methods below — the rest of the app stays unchanged.
class AIProvider(ABC):
    # Plain-text generation: returns the raw model output as a string.
    @abstractmethod
    async def generate(
        self,
        prompt: str,
    ) -> str:
        pass

    # JSON-structured generation: the provider returns a dict parsed from the
    # model's constrained JSON output, validated against the supplied schema.
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
    ) -> dict:
        pass
