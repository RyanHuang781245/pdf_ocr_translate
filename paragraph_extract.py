import os
import json
import re
import glob
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# ==============================
# 1️⃣ 初始化 (CPU 環境優化)
# ==============================
print("Loading BGE-M3 model on CPU...")
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("Model loaded.")

def is_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', str(text))

def is_english(text):
    return re.search(r'[A-Za-z]', str(text))

def bbox_vertical_match(box_zh, box_en):
    zh_x1, zh_y1, zh_x2, zh_y2 = box_zh
    en_x1, en_y1, en_x2, en_y2 = box_en
    if en_y1 < zh_y2: return False
    overlap = min(zh_x2, en_x2) - max(zh_x1, en_x1)
    zh_width = zh_x2 - zh_x1
    if overlap / zh_width < 0.5: return False
    if en_y1 - zh_y2 > 400: return False
    return True

# ==============================
# 2️⃣ 核心配對與回寫邏輯
# ==============================

def align_and_update_json(data):
    blocks = data.get("parsing_res_list", [])
    if not blocks:
        return data

    zh_list = []
    en_list = []
    
    # 用來強制排除的 label 清單
    exclude_labels = ["table", "paragraph_title"]

    # ---- 階段 A: 挑選出需要參與「語意配對」的 Block ----
    for i, block in enumerate(blocks):
        label = block.get("block_label", "")
        
        # 如果是 Table 或 Title，稍後統一設為 False，不參與向量運算
        if label in exclude_labels:
            continue
        
        text = block.get("block_content", "")
        bbox = block.get("block_bbox", [])
        
        if is_chinese(text):
            zh_list.append({"index": i, "text": text, "box": bbox})
        elif is_english(text):
            en_list.append({"index": i, "text": text, "box": bbox})

    matched_indices = set()
    pairs = []

    # ---- 階段 B: 執行向量配對 (僅針對 Content/Footer 等) ----
    if zh_list and en_list:
        zh_texts = [item["text"] for item in zh_list]
        en_texts = [item["text"] for item in en_list]
        
        # CPU 批次編碼
        zh_emb = model.encode(zh_texts, convert_to_tensor=True)
        en_emb = model.encode(en_texts, convert_to_tensor=True)
        cos_matrix = util.cos_sim(zh_emb, en_emb).numpy()

        for i, zh_item in enumerate(zh_list):
            best_score = 0
            best_en_idx = -1

            for j, en_item in enumerate(en_list):
                if not bbox_vertical_match(zh_item["box"], en_item["box"]):
                    continue

                score = cos_matrix[i][j]
                if score > best_score:
                    best_score = score
                    best_en_idx = j

            if best_score > 0.6:
                pairs.append({
                    "zh": zh_item["text"],
                    "en": en_list[best_en_idx]["text"],
                    "similarity": float(round(best_score, 4))
                })
                matched_indices.add(zh_item["index"])
                matched_indices.add(en_list[best_en_idx]["index"])

    # ---- 階段 C: 回寫 should_translate 標籤 ----
    for idx, block in enumerate(blocks):
        label = block.get("block_label", "")
        
        # 判斷邏輯：
        # 1. 如果 label 是 table 或 paragraph_title -> False
        # 2. 如果已經配對成功 -> False
        # 3. 其他 -> True
        if label in exclude_labels or idx in matched_indices:
            block["should_translate"] = False
        else:
            block["should_translate"] = True

    data["bilingual_pairs"] = pairs
    return data

# ==============================
# 3️⃣ 執行批次處理
# ==============================

def main():
    input_dir = "pp_json" 
    json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)

    print(f"開始批次處理 {len(json_files)} 個檔案...")

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            updated_data = align_and_update_json(data)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)
            
            print(f"成功回寫: {path}")

        except Exception as e:
            print(f"處理失敗 {path}: {e}")

if __name__ == "__main__":
    main()