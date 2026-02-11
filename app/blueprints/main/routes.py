from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from ...services import jobs, pipeline, state

main_bp = Blueprint(
    "main",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/main",
)


@main_bp.route("/", methods=["GET"], endpoint="index")
def index() -> str:
    return render_template("main/index.html")


@main_bp.route("/upload", methods=["POST"], endpoint="upload")
def upload() -> str:
    files = request.files.getlist("pdf")
    if not files or all(f.filename == "" for f in files):
        abort(400, "Missing PDF file.")

    state.JOB_ROOT.mkdir(parents=True, exist_ok=True)
    state.UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    dpi = 200
    start_page = int(request.form.get("start", 1))
    end_page_raw = request.form.get("end", "").strip()
    end_page = int(end_page_raw) if end_page_raw else None
    enable_translate = request.form.get("translate") == "on"
    translate_target_lang = request.form.get("target_lang", "en").strip() or "en"
    translate_model = request.form.get("model", state.AZURE_BATCH_MODEL).strip() or state.AZURE_BATCH_MODEL
    keep_lang = request.form.get("keep_lang", "all").strip().lower() or "all"
    if keep_lang not in {"all", "zh", "en"}:
        keep_lang = "all"

    for file in files:
        if not file or file.filename == "":
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in state.ALLOWED_EXTENSIONS:
            continue
        tmp_path = state.UPLOAD_ROOT / secure_filename(file.filename)
        file.save(tmp_path)
        display_name = secure_filename(Path(file.filename).stem) or "job"
        pipeline.enqueue_job_from_upload(
            tmp_path,
            display_name,
            dpi,
            start_page,
            end_page,
            translate_target_lang,
            translate_model,
            keep_lang,
            enable_translate,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    jobs.notify_jobs_update()

    return redirect(url_for(".index"))
