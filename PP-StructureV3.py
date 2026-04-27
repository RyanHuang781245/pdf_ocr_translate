import base64
import requests
import json
from pathlib import Path

API_URL = "https://writing-coordination-farm-approximately.trycloudflare.com/layout-parsing" # 服务URL

image_path = "output/pdf2img_outputs3/jpn_p0001.png"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
    "fileType": 1, # 文件类型，1表示图像文件
}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
md_dir = Path(f"api_test5")
md_dir.mkdir(exist_ok=True)
for i, res in enumerate(result["layoutParsingResults"]):
    # print(res["prunedResult"])
    pruned = res["prunedResult"]
    filename = f"{md_dir}/pruned_result_page_{i}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, ensure_ascii=False)
    (md_dir / "doc.md").write_text(res["markdown"]["text"], encoding="utf-8")
    for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"Markdown document saved at {md_dir / 'doc.md'}")
    for img_name, img in res["outputImages"].items():
        img_path = f"{md_dir}/{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")