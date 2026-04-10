from __future__ import annotations


class SimulatorApiError(RuntimeError):
    """Structured simulator error surfaced by HTTP or pipe transports."""

    def __init__(self, message: str, error_code: str | None = None,
                 status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
