import json
import numpy as np
from pathlib import Path
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

t0 = time.time()

from PaddleX.paddlex.inference.pipelines.layout_parsing.layout_objects import LayoutBlock

print("PaddleX import time:", time.time() - t0)

# ==============================
# FastAPI
# ==============================

app = FastAPI(title="Merge Service")


# ==============================
# 工具
# ==============================

def poly_to_box(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_inside(box, block):
    return not (
        box[2] < block[0] or
        box[0] > block[2] or
        box[3] < block[1] or
        box[1] > block[3]
    )


def inside_table(box, table_boxes):
    for t in table_boxes:
        if (
            box[0] >= t[0]
            and box[1] >= t[1]
            and box[2] <= t[2]
            and box[3] <= t[3]
        ):
            return True
    return False


def insert_pruned_result(data, pruned_result, after_key=None):

    if after_key is None:
        data["parsing_res_list"] = pruned_result
        return data

    new_data = {}

    for k, v in data.items():
        new_data[k] = v

        if k == after_key:
            new_data["parsing_res_list"] = pruned_result

    return new_data


# ==============================
# Merge 主流程（你的原程式）
# ==============================

def merge_keep_original_json(data):

    layout_blocks = data["layout_det_res"]["boxes"]
    ocr_res = data["overall_ocr_res"]
    tables = data["table_res_list"]

    rec_boxes = [poly_to_box(p) for p in ocr_res["rec_polys"]]
    rec_texts = ocr_res["rec_texts"]

    table_boxes = [
        b["coordinate"]
        for b in layout_blocks
        if b["label"] == "table"
    ]

    height = data.get("height")
    width = data.get("width")
    if not height or not width:
        input_path = data.get("input_path")
        if input_path:
            guess_name = Path(input_path).name
            m = __import__("re").search(r"_p(\\d+)", guess_name)
            if m:
                page_no = int(m.group(1))
                candidate = Path("out/images") / f"{Path(guess_name).stem}.png"
                if candidate.exists():
                    try:
                        from PIL import Image
                        with Image.open(candidate) as im:
                            width, height = im.size
                    except Exception:
                        pass
    if not height or not width:
        height, width = 2480, 3508

    dummy_img = np.zeros((height, width, 3), dtype=np.uint8)

    pruned_result = []
    table_idx = 0

    # ==========================
    # 建立 block list
    # ==========================
    for block in layout_blocks:

        label = block["label"]
        bbox = block["coordinate"]

        # ---------- TABLE ----------
        if label == "table":

            html = tables[table_idx]["pred_html"]
            table_idx += 1

            pruned_result.append({
                "block_label": "table",
                "block_content": html,
                "block_bbox": bbox,
            })

            continue

        # ---------- OCR 收集 ----------
        boxes = []
        texts = []

        for box, text in zip(rec_boxes, rec_texts):

            if inside_table(box, table_boxes):
                continue

            if is_inside(box, bbox):
                boxes.append(box)
                texts.append(text)

        if not boxes:
            continue

        # ---------- Paragraph merge ----------
        ocr_rec_res = {
            "boxes": boxes,
            "rec_texts": texts,
            "rec_labels": ["text"] * len(boxes),
        }

        lb = LayoutBlock(label=label, bbox=bbox)

        lb.update_text_content(
            image=dummy_img,
            ocr_rec_res=ocr_rec_res,
            text_rec_model=None,
        )

        pruned_result.append({
            "block_label": label,
            "block_content": lb.content.strip(),
            "block_bbox": bbox,
        })

    # 排序
    pruned_result.sort(
        key=lambda x: (x["block_bbox"][1], x["block_bbox"][0])
    )

    # 合併回原 JSON
    data = insert_pruned_result(
        data,
        pruned_result,
        after_key="doc_preprocessor_res",
    )

    return data


# ==============================
# API：Merge + 存檔
# ==============================

@app.post("/merge-and-save")
async def merge_and_save(file: UploadFile = File(...)):

    t1 = time.time()

    # 讀 JSON
    content = await file.read()
    data = json.loads(content.decode("utf-8"))

    # Merge
    merged = merge_keep_original_json(data)

    return JSONResponse({
        "status": "success",
        "merged": merged,
        "time_cost": round(time.time() - t1, 3),
    })


# ==============================
# Root 測試
# ==============================

@app.get("/")
def root():
    return {"status": "merge service running"}
