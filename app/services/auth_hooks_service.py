from __future__ import annotations

from flask import jsonify, redirect, request, url_for
from flask_login import current_user

from . import auth_store
from .authz_service import sanitize_next_url


_PUBLIC_ENDPOINTS = {
    "auth.login",
    "auth.logout",
    "static",
}


def register_auth_context(app) -> None:
    @app.context_processor
    def inject_auth_context():
        user = current_user if current_user.is_authenticated else None
        current_user_display_name = None
        if user is not None:
            try:
                snapshot = auth_store.get_local_user_snapshot(getattr(user, "work_id", ""))
            except Exception:
                snapshot = None
            current_user_display_name = (
                snapshot.display_name
                if snapshot and getattr(snapshot, "display_name", "")
                else getattr(user, "display_name", None)
            )
        return {
            "auth_enabled": app.config.get("AUTH_ENABLED", False),
            "current_user": user,
            "current_user_display_name": current_user_display_name,
        }



def enforce_login_for_request(app):
    if not app.config.get("AUTH_ENABLED", False):
        return None

    if request.endpoint in _PUBLIC_ENDPOINTS or request.endpoint is None:
        return None

    if current_user.is_authenticated:
        return None

    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Authentication required."}), 401

    return redirect(url_for("auth.login", next=sanitize_next_url(request.full_path)))
