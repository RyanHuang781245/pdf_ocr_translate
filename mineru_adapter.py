import json
import subprocess
from pathlib import Path

def run_mineru(pdf_path: str, out_json_path: str) -> dict:
    """
    期待 MinerU 產出結構化 JSON，最終回傳 dict。
    你只要把 cmd 換成你環境中 MinerU 的真實指令即可。
    """
    pdf_path = str(Path(pdf_path).resolve())
    out_json_path = str(Path(out_json_path).resolve())

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: 請依你的 MinerU 安裝方式調整
    # 範例（假設 mineru 支援 --input/--output/--format json）
    cmd = ["mineru", "--input", pdf_path, "--output", out_json_path, "--format", "json"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("找不到 mineru 指令，請確認已安裝並在 PATH 中") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MinerU 執行失敗：{e.stderr}") from e

    with open(out_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def mineru_json_to_blocks(mineru_json: dict) -> list[dict]:
    """
    把 MinerU JSON 轉成 blocks。
    期望格式（你可依 MinerU 真實輸出調整 mapping）：
      pages: [
        { page_no: 1, width:..., height:..., blocks:[
           {type:'paragraph', bbox:[x0,y0,x1,y1], text:'...'}
        ]}
      ]
    """
    out = []
    pages = mineru_json.get("pages", [])
    for p in pages:
        page_no = int(p.get("page_no", 1))
        for i, b in enumerate(p.get("blocks", [])):
            bbox = b.get("bbox") or [0, 0, 0, 0]
            out.append({
                "page_no": page_no,
                "reading_order": i,
                "block_type": b.get("type", "paragraph"),
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
                "text": (b.get("text") or "").strip()
            })
    return out
