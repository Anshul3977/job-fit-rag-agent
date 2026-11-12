"""Structured application errors. Handlers turn these into JSON bodies."""


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or {}


class ExtractError(Exception):
    """The file was the right type but text could not be read."""
