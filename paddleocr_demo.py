from paddleocr import PaddleOCR
# 初始化 PaddleOCR 執行個體
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 對範例圖片執行 OCR 推論
result = ocr.predict(
    input="test_page_0.png")
    
# 將結果視覺化並儲存為 JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")