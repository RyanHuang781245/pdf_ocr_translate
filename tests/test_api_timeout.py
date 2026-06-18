from __future__ import annotations

import json

from app.services import state


# API timeout 驗測標準（route 層）：
# 1. 下游 OCR / 翻譯 API timeout 時，Flask route 應回傳錯誤狀態與可判斷的 JSON。
# 2. timeout 不應把半成品寫回 edits.json，避免覆蓋使用者既有編輯。
# 3. timeout 發生在後續 PDF 輸出前，因此不應進入成功流程或呼叫 PDF 重繪。
# 4. 覆蓋前端會觸發的補翻入口：OCR preview、單框補翻、多框補翻、區域補翻、詞彙補翻。


def _write_batch_config(job_dir):
    (job_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "document_mode": "general_force",
                "target_lang": "en",
                "model": "fake-model",
                "system_prompt": "translate",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_edits(job_dir, edits):
    (job_dir / "edits.json").write_text(
        json.dumps(edits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_edits(job_dir):
    return json.loads((job_dir / "edits.json").read_text(encoding="utf-8"))


def _fail_if_pdf_render_is_called(monkeypatch):
    def fail_render(*args, **kwargs):
        raise AssertionError("PDF render should not be called after upstream timeout.")

    monkeypatch.setattr("app.blueprints.api.routes.ocr.apply_edits_to_pdf", fail_render)


def test_retranslate_box_timeout_returns_error_without_mutating_edits(client, tmp_path, monkeypatch):
    # 覆蓋情境：選取單一文字框後呼叫翻譯 API timeout。
    # 驗證契約：API 回 500 + ok=false，且目標文字框不會被改成半完成翻譯結果。
    job_id = "8" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 200001,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "Old translation",
                        "auto_generated": True,
                        "tm_source_text": "舊原文",
                    }
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)

    def fake_translate_timeout(texts, **kwargs):
        raise TimeoutError("Request timed out. (read timeout=3.5s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.batch.translate_texts_for_region",
        fake_translate_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/retranslate-box",
        json={
            "page_index_0based": 0,
            "box_id": 200001,
            "source_text": "修正後原文",
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=3.5s)"}
    assert _read_edits(job_dir) == original_edits


def test_retranslate_boxes_timeout_returns_error_without_mutating_edits(client, tmp_path, monkeypatch):
    # 覆蓋情境：批次選取多個文字框後呼叫翻譯 API timeout。
    # 驗證契約：任何目標框都不應被部分更新，整份 edits.json 必須維持原狀。
    job_id = "6" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 200001,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "Box one",
                        "auto_generated": False,
                    },
                    {
                        "id": 200002,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 40, "w": 80, "h": 20},
                        "text": "Box two",
                        "auto_generated": False,
                    },
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)

    def fake_translate_timeout(texts, **kwargs):
        raise TimeoutError("Request timed out. (read timeout=3.5s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.batch.translate_texts_for_region",
        fake_translate_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/retranslate-boxes",
        json={
            "targets": [
                {
                    "page_index_0based": 0,
                    "box_id": 200001,
                    "source_text": "第一段",
                },
                {
                    "page_index_0based": 0,
                    "box_id": 200002,
                    "source_text": "第二段",
                },
            ]
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=3.5s)"}
    assert _read_edits(job_dir) == original_edits


def test_region_ocr_preview_timeout_returns_error_without_creating_edits(client, tmp_path, monkeypatch):
    # 覆蓋情境：前端只預覽框選區域 OCR，OCR API timeout。
    # 驗證契約：API 回 500 + ok=false，且 preview 不應建立 edits.json 或觸發輸出流程。
    job_id = "5" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _fail_if_pdf_render_is_called(monkeypatch)

    def fake_region_ocr_timeout(current_job_dir, page_idx, bbox):
        raise TimeoutError("Request timed out. (read timeout=1s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.ocr.run_region_ocr",
        fake_region_ocr_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/region-ocr-preview",
        json={
            "page_index_0based": 0,
            "bbox": {"x": 0, "y": 0, "w": 120, "h": 60},
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=1s)"}
    assert not (job_dir / "edits.json").exists()


def test_retranslate_region_ocr_timeout_returns_error_without_mutating_edits(client, tmp_path, monkeypatch):
    # 覆蓋情境：區域補翻需要先做區域 OCR，但 OCR API timeout。
    # 驗證契約：API 回 500 + ok=false，且原本的自動文字框不會被標 deleted 或新增補翻框。
    job_id = "7" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 100000,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "old auto",
                        "auto_generated": True,
                    }
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)

    def fake_region_ocr_timeout(current_job_dir, page_idx, bbox):
        raise TimeoutError("Request timed out. (read timeout=1s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.ocr.run_region_ocr",
        fake_region_ocr_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/retranslate-region",
        json={
            "page_index_0based": 0,
            "bbox": {"x": 0, "y": 0, "w": 120, "h": 60},
            "replace_existing": True,
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=1s)"}
    assert _read_edits(job_dir) == original_edits


def test_retranslate_region_translation_timeout_after_ocr_does_not_delete_or_add_boxes(client, tmp_path, monkeypatch):
    # 覆蓋情境：區域 OCR 成功後，翻譯 API timeout。
    # 驗證契約：replace_existing=True 也不能先刪舊框；沒有翻譯結果就不得新增補翻框。
    job_id = "4" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 100000,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "old auto",
                        "auto_generated": True,
                    }
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)

    monkeypatch.setattr(
        "app.blueprints.api.routes.ocr.run_region_ocr",
        lambda current_job_dir, page_idx, bbox: {
            "page_index_0based": page_idx,
            "region_bbox": bbox,
            "rec_polys": [[[15, 15], [65, 15], [65, 25], [15, 25]]],
            "rec_texts": ["補翻來源"],
            "rec_scores": [0.99],
        },
    )

    def fake_translate_timeout(texts, **kwargs):
        raise TimeoutError("Request timed out. (read timeout=3.5s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.batch.translate_texts_for_region",
        fake_translate_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/retranslate-region",
        json={
            "page_index_0based": 0,
            "bbox": {"x": 0, "y": 0, "w": 120, "h": 60},
            "replace_existing": True,
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=3.5s)"}
    assert _read_edits(job_dir) == original_edits


def test_retranslate_region_source_text_translation_timeout_does_not_delete_or_add_boxes(client, tmp_path, monkeypatch):
    # 覆蓋情境：前端已提供 source_text，區域補翻略過 OCR 後翻譯 API timeout。
    # 驗證契約：即使不走 OCR 分支，timeout 仍不得刪除既有框或新增補翻框。
    job_id = "3" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 100000,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "old auto",
                        "auto_generated": True,
                    }
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)

    def fake_translate_timeout(texts, **kwargs):
        raise TimeoutError("Request timed out. (read timeout=3.5s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.batch.translate_texts_for_region",
        fake_translate_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/retranslate-region",
        json={
            "page_index_0based": 0,
            "bbox": {"x": 0, "y": 0, "w": 120, "h": 60},
            "merged_bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
            "source_text": "前端已取得的來源文字",
            "replace_existing": True,
        },
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=3.5s)"}
    assert _read_edits(job_dir) == original_edits


def test_glossary_retranslate_timeout_returns_error_without_mutating_matching_boxes(client, tmp_path, monkeypatch):
    # 覆蓋情境：詞彙表更新後，針對含該詞彙的既有框重新翻譯時 timeout。
    # 驗證契約：匹配到的文字框不能被部分更新，edits.json 必須維持原狀。
    job_id = "2" * 32
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(state, "JOB_ROOT", tmp_path / "jobs")
    _write_batch_config(job_dir)
    _fail_if_pdf_render_is_called(monkeypatch)

    original_edits = {
        "pages": [
            {
                "page_index_0based": 0,
                "boxes": [
                    {
                        "id": 200001,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "text": "Old translation",
                        "auto_generated": True,
                        "tm_source_text": "這段包含關鍵詞",
                        "tm_source_normalized": "這段包含關鍵詞",
                    },
                    {
                        "id": 200002,
                        "deleted": False,
                        "bbox": {"x": 10, "y": 40, "w": 80, "h": 20},
                        "text": "Unmatched translation",
                        "auto_generated": True,
                        "tm_source_text": "另一段",
                    },
                ],
            }
        ]
    }
    _write_edits(job_dir, original_edits)
    monkeypatch.setattr("app.blueprints.api.routes.glossary.load_combined_glossary", lambda: [])

    def fake_translate_timeout(texts, **kwargs):
        raise TimeoutError("Request timed out. (read timeout=3.5s)")

    monkeypatch.setattr(
        "app.blueprints.api.routes.batch.translate_texts_for_region",
        fake_translate_timeout,
    )

    resp = client.post(
        f"/api/job/{job_id}/glossary-retranslate",
        json={"cn": "關鍵詞"},
    )

    assert resp.status_code == 500
    assert resp.get_json() == {"ok": False, "error": "Request timed out. (read timeout=3.5s)"}
    assert _read_edits(job_dir) == original_edits
