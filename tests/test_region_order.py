from app.services.ocr import build_region_rows


def test_build_region_rows_orders_left_to_right_within_each_row():
    rec_polys = [
        [[60, 10], [90, 10], [90, 30], [60, 30]],
        [[10, 10], [50, 10], [50, 30], [10, 30]],
        [[20, 40], [80, 40], [80, 60], [20, 60]],
    ]
    rec_texts = ["名稱", "專案", "會議記錄單"]

    rows = build_region_rows(rec_polys, rec_texts)

    assert rows == ["專案名稱", "會議記錄單"]
