import json
import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for

from dotenv import load_dotenv
from db import init_db, get_conn
from mineru_adapter import run_mineru, mineru_json_to_blocks
from gpt_translate import translate_text
from pdf_export import export_translated_pdf
from pypdf import PdfReader

load_dotenv()

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MINERU_DIR = DATA_DIR / "mineru"

for d in [UPLOAD_DIR, OUTPUT_DIR, MINERU_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

@app.get("/")
def index():
    conn = get_conn()
    docs = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("index.html", docs=docs)

@app.post("/upload")
def upload():
    f = request.files.get("pdf")
    if not f or not f.filename.lower().endswith(".pdf"):
        return "請上傳 PDF", 400

    save_path = UPLOAD_DIR / f"{Path(f.filename).stem}_{os.urandom(4).hex()}.pdf"
    f.save(save_path)

    reader = PdfReader(str(save_path))
    page_count = len(reader.pages)

    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO documents(filename, upload_path, page_count) VALUES(?,?,?)",
        (f.filename, str(save_path), page_count)
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()

    return redirect(url_for("editor", doc_id=doc_id))

@app.get("/doc/<int:doc_id>")
def editor(doc_id: int):
    return render_template("editor.html", doc_id=doc_id)

@app.post("/api/doc/<int:doc_id>/parse_and_translate")
def parse_and_translate(doc_id: int):
    """
    同步：MinerU 解析 + GPT 翻譯，寫入 blocks/translations
    """
    conn = get_conn()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        return jsonify({"error": "doc not found"}), 404

    # 清掉舊資料（方便重跑）
    conn.execute("DELETE FROM translations WHERE block_id IN (SELECT id FROM blocks WHERE document_id=?)", (doc_id,))
    conn.execute("DELETE FROM blocks WHERE document_id=?", (doc_id,))
    conn.execute("DELETE FROM annotations WHERE document_id=?", (doc_id,))

    mineru_out = MINERU_DIR / f"doc_{doc_id}.json"
    mineru_json = run_mineru(doc["upload_path"], str(mineru_out))
    blocks = mineru_json_to_blocks(mineru_json)

    # 寫 blocks
    for b in blocks:
        if not b["text"]:
            continue
        cur = conn.execute(
            """INSERT INTO blocks(document_id,page_no,block_type,x0,y0,x1,y1,text_original,reading_order)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (doc_id, b["page_no"], b["block_type"], b["x0"], b["y0"], b["x1"], b["y1"], b["text"], b["reading_order"])
        )
        block_id = cur.lastrowid

        # 翻譯（同步）
        zh = translate_text(b["text"], target_lang="zh-Hant")
        conn.execute(
            "INSERT INTO translations(block_id,text_translated) VALUES(?,?)",
            (block_id, zh)
        )

    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.get("/api/doc/<int:doc_id>/meta")
def doc_meta(doc_id: int):
    conn = get_conn()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        return jsonify({"error": "doc not found"}), 404
    return jsonify({"id": doc["id"], "page_count": doc["page_count"]})

@app.get("/api/doc/<int:doc_id>/pdf")
def doc_pdf(doc_id: int):
    conn = get_conn()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        return "not found", 404
    return send_file(doc["upload_path"], mimetype="application/pdf")

@app.get("/api/doc/<int:doc_id>/page/<int:page_no>/blocks")
def page_blocks(doc_id: int, page_no: int):
    conn = get_conn()
    rows = conn.execute(
        """SELECT b.*, t.text_translated
           FROM blocks b
           LEFT JOIN translations t ON t.block_id=b.id
           WHERE b.document_id=? AND b.page_no=?
           ORDER BY b.reading_order ASC""",
        (doc_id, page_no)
    ).fetchall()
    conn.close()

    blocks = []
    for r in rows:
        blocks.append({
            "block_id": r["id"],
            "type": r["block_type"],
            "bbox": [r["x0"], r["y0"], r["x1"], r["y1"]],
            "text_original": r["text_original"],
            "text_translated": r["text_translated"] or ""
        })
    return jsonify({"blocks": blocks})

@app.get("/api/doc/<int:doc_id>/page/<int:page_no>/annotations")
def get_annotations(doc_id: int, page_no: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT canvas_json FROM annotations WHERE document_id=? AND page_no=?",
        (doc_id, page_no)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"canvas": None})
    return jsonify({"canvas": json.loads(row["canvas_json"])})

@app.put("/api/doc/<int:doc_id>/page/<int:page_no>/annotations")
def save_annotations(doc_id: int, page_no: int):
    data = request.get_json(force=True)
    canvas = data.get("canvas")
    if canvas is None:
        return jsonify({"error": "missing canvas"}), 400

    conn = get_conn()
    conn.execute(
        """INSERT INTO annotations(document_id,page_no,canvas_json)
           VALUES(?,?,?)
           ON CONFLICT(document_id,page_no) DO UPDATE SET
             canvas_json=excluded.canvas_json,
             updated_at=datetime('now')""",
        (doc_id, page_no, json.dumps(canvas, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.post("/api/doc/<int:doc_id>/export")
def export_pdf(doc_id: int):
    conn = get_conn()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        return jsonify({"error": "doc not found"}), 404

    rows = conn.execute(
        "SELECT page_no, canvas_json FROM annotations WHERE document_id=?",
        (doc_id,)
    ).fetchall()
    conn.close()

    annotations_by_page = {}
    for r in rows:
        annotations_by_page[int(r["page_no"])] = json.loads(r["canvas_json"])

    out_path = OUTPUT_DIR / f"doc_{doc_id}_translated.pdf"
    export_translated_pdf(doc["upload_path"], annotations_by_page, str(out_path))
    return jsonify({"ok": True, "download_url": url_for("download_export", doc_id=doc_id)})

@app.get("/doc/<int:doc_id>/download")
def download_export(doc_id: int):
    out_path = OUTPUT_DIR / f"doc_{doc_id}_translated.pdf"
    if not out_path.exists():
        return "尚未匯出", 404
    return send_file(str(out_path), as_attachment=True, download_name=out_path.name)

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
