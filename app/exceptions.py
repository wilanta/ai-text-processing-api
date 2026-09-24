# Exception hierarchy for AI-service failures.
# All classes inherit from AIServiceError so callers can catch the base class
# to handle every AI-related failure in a single branch.


class AIServiceError(Exception):
    """Base exception for AI service errors — the root of the hierarchy."""


class AIConnectionError(AIServiceError):
    """Raised when the HTTP client cannot reach the AI provider (network down, wrong URL, port closed)."""


class AITimeoutError(AIServiceError):
    """Raised when the AI provider does not respond within the configured timeout window."""


class AIResponseError(AIServiceError):
    """Raised when the provider responds successfully but returns malformed or incomplete data."""
