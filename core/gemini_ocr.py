import io
import requests
import json
import base64

from core.pdf_loader import PDFLoader


class GeminiOCR:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.aifuwu.icu/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def recognize_stream(self, system_prompt: str, prompt: str, images):
        parts = []

        if not isinstance(images, list):
            images = [images]

        # === 添加图片处理逻辑 ===
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": image_data
                }
            })

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

        response = self.session.post(self.url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode('utf-8').strip()

            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                if json_str == "[DONE]":
                    break

                if not json_str:
                    continue

                chunk = json.loads(json_str)
                candidates = chunk.get('candidates', [])
                if candidates:
                    content_parts = candidates[0].get('content', {}).get('parts', [])
                    for part in content_parts:
                        if 'text' in part and not part.get('thought'):
                            yield part['text']


if __name__ == "__main__":
    API_KEY = "sk-CBGqz24SA062pFwmPrImsUPMs19uS2RbfOg5OFCAzgMhBFzO"
    loader = PDFLoader("../串起粒粒的宝珠：摩诃止观导读.pdf")
    page_img = loader.get_page(390)

    system_prompt = (
        "你是一个影印PDF转录md文档的专家。"
        "你会把页底的注脚注释放到正文对应位置的后面。"
        "你会忽略页眉页脚，只关注有价值的部分。"
        "你只会输出识别内容。"
    )
    prompt = "这是关于[佛法,摩诃止观]的文档片段图。请将这张图片转录为Markdown格式。"

    ocr = GeminiOCR(API_KEY)

    for chunk in ocr.recognize_stream(system_prompt, prompt, page_img):
        print(chunk, end="", flush=True)