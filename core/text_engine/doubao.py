"""Doubao (火山引擎) 实现类。"""
import os
from openai import OpenAI
from .base import BaseTextModel


class DoubaoText(BaseTextModel):
    """使用火山引擎 (豆包) 模型进行文本处理。"""
    def __init__(self, model_name: str = "doubao-seed-2-0-pro-260215", temperature: float = 0.3, top_p: float = 0.95):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 ARK_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=600.0,
        )

    def generate_stream(self, system_prompt: str, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
