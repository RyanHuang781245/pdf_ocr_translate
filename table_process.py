import json
import re
from pathlib import Path


# =========================================================
# 語言判斷
# =========================================================

def is_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text)


def is_english(text):
    return re.search(r'[A-Za-z]', text)


def is_mixed(text):
    return is_chinese(text) and is_english(text)


def is_code_like(text):
    return re.fullmatch(r'[A-Za-z0-9\-_/\. %]+', text)


def should_translate(text):

    text = text.strip()

    if not text:
        return False

    if is_mixed(text):
        return False

    if is_code_like(text):
        return False

    if is_chinese(text):
        return True

    return False


# =========================================================
# 座標工具
# =========================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)


def intersection_area(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 < x1 or y2 < y1:
        return 0

    return (x2-x1)*(y2-y1)


def box_area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def overlap_ratio(cell_box, rec_box):

    inter = intersection_area(cell_box, rec_box)
    area = box_area(rec_box)

    if area == 0:
        return 0

    return inter / area


def center_inside(cell_box, rec_box):

    cx1, cy1, cx2, cy2 = cell_box
    rx, ry = box_center(rec_box)

    return (
        cx1 <= rx <= cx2 and
        cy1 <= ry <= cy2
    )


def is_inside_cell(cell_box, rec_box,
                   overlap_thresh=0.7):

    if center_inside(cell_box, rec_box):
        return True

    if overlap_ratio(cell_box, rec_box) > overlap_thresh:
        return True

    return False


# =========================================================
# 排序
# =========================================================

def sort_by_reading_order(texts_with_box):

    def center(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2 , (y1+y2)/2)

    return sorted(
        texts_with_box,
        key=lambda x: (center(x[1])[1], center(x[1])[0])
    )


# =========================================================
# Cell 合併
# =========================================================

def merge_cell_text(cell_box, rec_texts, rec_boxes):

    texts_in_cell = []

    for txt, box in zip(rec_texts, rec_boxes):

        if is_inside_cell(cell_box, box):
            texts_in_cell.append((txt, box))

    if not texts_in_cell:
        return ""

    texts_in_cell = sort_by_reading_order(texts_in_cell)

    merged = "".join([t[0] for t in texts_in_cell])

    return merged.strip()


# =========================================================
# 新增 merged_cells 欄位
# =========================================================

def add_merged_cells_field(data):

    for table_idx, table in enumerate(data["table_res_list"]):

        print(f"\n處理 Table {table_idx+1}")

        cell_boxes = table["cell_box_list"]
        rec_texts = table["table_ocr_pred"]["rec_texts"]
        rec_boxes = table["table_ocr_pred"]["rec_boxes"]

        merged_cells = []

        for cell_idx, cell_box in enumerate(cell_boxes):

            merged_text = merge_cell_text(
                cell_box,
                rec_texts,
                rec_boxes
            )

            translate_flag = should_translate(merged_text)

            merged_cells.append({
                "cell_box": cell_box,
                "merged_text": merged_text,
                "should_translate": translate_flag
            })

            print(f"[Cell {cell_idx+1}] {merged_text} → {translate_flag}")

        # 新增欄位（不覆蓋原 OCR）
        table["merged_cells"] = merged_cells

    return data


# =========================================================
# 主程式
# =========================================================

if __name__ == "__main__":

    input_path = Path("out/jobs/f8663368c9b7476683c44ac7cb399ea3/pp_json/f8663368c9b7476683c44ac7cb399ea3_p0002.json")          # OCR JSON
    output_path = Path("table_merged.json")  # 輸出

    data = json.loads(input_path.read_text(encoding="utf-8"))

    new_data = add_merged_cells_field(data)

    output_path.write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n完成 → table_merged.json")
