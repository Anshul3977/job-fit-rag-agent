"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.db import init_db
from app.errors import AppError
from app.logging_config import RequestContextMiddleware, configure_logging, request_id_var


def _error_body(status_code: int, error_code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "details": details or {},
            "request_id": request_id_var.get(),
        },
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    if settings.resolved_vector_backend == "pgvector":
        from app.runtime import get_store

        store = get_store()
        setup = getattr(store, "setup", None)
        if setup is not None:
            setup()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Job Fit RAG Agent",
        version=__version__,
        summary="Score a resume against a job post and cite the resume lines that support the score.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list or ["http://127.0.0.1:5174"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)

    @app.exception_handler(AppError)
    async def app_error(_request, exc: AppError):
        return _error_body(exc.status_code, exc.error_code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc: RequestValidationError):
        fields = []
        for err in exc.errors():
            loc = [str(part) for part in err.get("loc", []) if part not in {"body", "query", "form"}]
            fields.append({"field": ".".join(loc) or "request", "message": err.get("msg", "invalid")})
        return _error_body(
            422,
            "VALIDATION_ERROR",
            "Request validation failed.",
            {"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc: StarletteHTTPException):
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error_body(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def unhandled(_request, _exc: Exception):
        return _error_body(500, "INTERNAL_ERROR", "Unexpected server error.")

    dist = Path(settings.frontend_dist)
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        page = dist / "index.html"
        if page.is_file():
            return FileResponse(page)
        return {
            "service": settings.app_name,
            "version": settings.version,
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        icon = dist / "favicon.svg"
        if icon.is_file():
            return FileResponse(icon, media_type="image/svg+xml")
        raise StarletteHTTPException(status_code=404, detail="Not found.")

    return app


app = create_app()
