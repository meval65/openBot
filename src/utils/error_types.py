class BotSystemError(Exception):
    """Base exception class for all bot system errors."""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause

class DatabaseError(BotSystemError):
    """Raised when a database operation fails."""
    pass

class MemoryError(BotSystemError):
    """Raised when a memory operation (retrieval, storage, deduplication) fails."""
    pass

class LLMGenerationError(BotSystemError):
    """Raised when generating a response from an LLM fails."""
    pass

class ConfigurationError(BotSystemError):
    """Raised when there is an issue with the system configuration."""
    pass

class IntegrationError(BotSystemError):
    """Raised when an external API or integration (e.g. OpenRouter, Meteosource) fails."""
    pass
