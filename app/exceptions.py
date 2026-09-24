class AIServiceError(Exception):
    """Base exception for AI service errors."""


class AIConnectionError(AIServiceError):
    """Cannot connect to AI provider."""


class AITimeoutError(AIServiceError):
    """AI provider took too long to respond."""


class AIResponseError(AIServiceError):
    """AI provider returned an invalid response."""
