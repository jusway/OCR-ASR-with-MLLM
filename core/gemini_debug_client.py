import io

import requests
import json
import base64
import os
from pathlib import Path

from core.pdf_loader import PDFLoader


class GeminiDebugClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 尝试添加 ?alt=sse 参数来强制开启 Server-Sent Events 流式模式
        self.url = "https://api.aifuwu.icu/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片文件编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def get_mime_type(self, image_path: str) -> str:
        """根据文件扩展名获取 MIME 类型"""
        extension = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(extension, "image/jpeg")

    def chat(self, system_prompt: str, prompt: str, images):
        parts = []

        if not isinstance(images, list):
            images = [images]

        # === 添加图片处理逻辑 ===
        for idx, img in enumerate(images):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mime_type = "image/png"

            parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": image_data
                }
            })
            print(f"✅ 已加载图片 {idx + 1} (image/png)")

        # 添加问题文本
        parts.append({"text": prompt})

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 1,
                "topP": 1,
                "thinkingConfig": {
                    "includeThoughts": True,
                    "thinkingBudget": 26240
                }
            }
        }

        print(f"--- 正在发送 {len(images)} 张图片和问题... ---")
        response = requests.post(self.url, headers=self.headers, json=payload, stream=True)
        response.raise_for_status()
        print("--- 连接成功，开始接收数据 ---")

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode('utf-8').strip()

            print(f"[原始数据]: {decoded_line}")

            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                if json_str == "[DONE]":
                    break
            else:
                json_str = decoded_line

            if json_str in ['[', ']', ','] or json_str == '':
                continue
            if json_str.startswith(','):
                json_str = json_str[1:]

            if json_str.startswith('['):
                pass

            chunk = json.loads(json_str)

            candidates = chunk.get('candidates', [])
            if candidates:
                content_parts = candidates[0].get('content', {}).get('parts', [])
                for part in content_parts:
                    if 'text' in part:
                        print(f"✅ 解析成功 ({len(part['text'])} chars): {part['text'][:20]}...")

        print("\n--- 回复结束 ---")

if __name__ == "__main__":
    API_KEY = "sk-CBGqz24SA062pFwmPrImsUPMs19uS2RbfOg5OFCAzgMhBFzO"

    # 示例本地图片路径
    loader=PDFLoader("../串起粒粒的宝珠：摩诃止观导读.pdf")
    page_img = loader.get_page(390)
    print(type(page_img))
    system_prompt=("你是一个影印PDF转录md文档的专家。"
                   "你会把页底的注脚注释放到正文对应位置的后面。"
                   "你会忽略页眉页脚，只关注有价值的部分。"
                   "你只会输出识别内容。"
                   )
    prompt = "这是关于[佛法,摩诃止观]的文档片段图。请将这张图片转录为Markdown格式。"

    client = GeminiDebugClient(API_KEY)
    client.chat(system_prompt,prompt, page_img)






