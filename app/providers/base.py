from abc import ABC, abstractmethod


class AIProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
    ) -> str:
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
    ) -> dict:
        pass
