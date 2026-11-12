"""JSON request logs with a request id. Bodies are never logged."""

import contextvars
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not getattr(root, "_jobfit_configured", False):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.handlers = [handler]
        root._jobfit_configured = True  # type: ignore[attr-defined]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, "request_id": request_id_var.get(), **fields}
    logging.getLogger("jobfit").info(json.dumps(payload, default=str))


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        request_id = incoming.strip()[:64] if incoming else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                "request",
                method=request.method,
                path=request.url.path,
                status=status,
                latency_ms=latency_ms,
            )
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response
