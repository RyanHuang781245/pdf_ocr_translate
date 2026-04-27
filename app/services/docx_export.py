from __future__ import annotations

import subprocess
from pathlib import Path


def export_markdown_to_docx(markdown_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "pandoc",
            markdown_path.as_posix(),
            "-f",
            "markdown",
            "-t",
            "docx",
            "-o",
            out_path.as_posix(),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pandoc export failed")
    return out_path
