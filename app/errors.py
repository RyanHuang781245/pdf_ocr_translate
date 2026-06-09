from __future__ import annotations

from flask import request
from werkzeug.exceptions import HTTPException

from .services import audit_service


def register_error_handlers(app) -> None:
    def handle_http_exception(error: HTTPException):
        if int(getattr(error, "code", 500) or 500) >= 500:
            audit_service.record_system_error(
                "web.http_exception",
                getattr(error, "description", "HTTP exception"),
                exc=error,
                detail={
                    "path": request.path,
                    "method": request.method,
                    "endpoint": request.endpoint or "",
                },
            )
        return error

    def handle_unhandled_exception(error: Exception):
        if app.config.get("TESTING"):
            raise error
        audit_service.record_system_error(
            "web.unhandled_exception",
            "Unhandled web exception",
            exc=error,
            detail={
                "path": request.path,
                "method": request.method,
                "query_string": request.query_string.decode("utf-8", errors="ignore"),
                "endpoint": request.endpoint or "",
            },
        )
        return "Internal Server Error", 500

    app.register_error_handler(HTTPException, handle_http_exception)
    app.register_error_handler(Exception, handle_unhandled_exception)
