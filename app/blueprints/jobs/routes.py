from __future__ import annotations

from flask import Blueprint, abort, send_from_directory

from ...services import jobs

jobs_bp = Blueprint(
    "jobs",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/jobs",
    url_prefix="",
)


@jobs_bp.route("/jobs/<job_id>/<path:filename>", methods=["GET"], endpoint="job_file")
def job_file(job_id: str, filename: str):
    if not jobs.safe_job_id(job_id):
        abort(404)
    job_dir = jobs.job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    return send_from_directory(job_dir, filename)
