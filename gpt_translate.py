import os
from openai import OpenAI

client = OpenAI()

def translate_text(text: str, target_lang: str = "zh-Hant") -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not text.strip():
        return ""

    instructions = (
        "你是專業翻譯。請忠實翻譯，保留：代碼、單位、標準編號、變數名、專有名詞。"
        "不要自行增刪內容。輸出只要譯文。"
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=f"請翻譯成 {target_lang}：\n\n{text}"
    )
    return (resp.output_text or "").strip()
