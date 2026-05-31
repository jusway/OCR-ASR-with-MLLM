"""MiMo 实现类 (OpenAI 兼容接口)。"""
import os
import json
import requests
from .base import BaseTextModel


class MiMoText(BaseTextModel):
    """使用 MiMo 模型进行文本处理。"""
    def __init__(self, model_name: str = "mimo-v2-pro", temperature: float = 0.3, top_p: float = 0.95,
                 api_key: str = None):
        super().__init__(model_name, temperature, top_p)
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        self.base_url = "https://api.xiaomimimo.com/v1/chat/completions"
        if not self.api_key:
            raise ValueError("未找到 MIMO_API_KEY，请设置环境变量或在初始化时传入。")

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application / json"
        }

    def _build_payload(self, system_prompt: str, prompt: str, stream: bool):
        return {
            "model": self.model_name,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
            "thinking": {"type": "disabled"}
        }

    def generate_stream(self, system_prompt: str, prompt: str):
        payload = self._build_payload(system_prompt, prompt, stream=True)
        response = requests.post(self.base_url, headers=self._get_headers(), json=payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    content = line_str[6:]
                    if content == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(content)
                        delta = chunk_data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
                    except json.JSONDecodeError:
                        continue
