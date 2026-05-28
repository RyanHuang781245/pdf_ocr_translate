from __future__ import annotations

from flask import Blueprint

admin_bp = Blueprint(
    "admin",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/admin",
    url_prefix="/admin",
)
