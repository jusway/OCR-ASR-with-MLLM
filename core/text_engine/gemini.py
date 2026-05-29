"""Gemini 实现类。"""
import time
from google import genai
from google.genai import types, errors
from core.utils import Color
from .base import BaseTextModel


class GeminiText(BaseTextModel):
    def __init__(self, model_name: str = "gemini-3.1-pro-preview", temperature: float = 1.0, top_p: float = 0.95):
        super().__init__(model_name, temperature, top_p)
        self.client = genai.Client(
            http_options={
                'timeout': None,
                'client_args': {'proxy': None, 'trust_env': False}
            }
        )

    def generate(self, system_prompt: str, prompt: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
            except errors.APIError as e:
                if attempt < max_retries - 1:
                    print(f"{Color.RED}⚠️ Gemini API 调用失败 (尝试 {attempt + 1}/{max_retries}): {repr(e)}")
                    print(f"{Color.RED}等待 3 秒后重试...\033[0m")
                    time.sleep(3)
                else:
                    raise

    def generate_stream(self, system_prompt: str, prompt: str):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                break
            except errors.APIError as e:
                if attempt < max_retries - 1:
                    print(f"\n{Color.RED}⚠️ Gemini API 流式调用失败 (尝试 {attempt + 1}/{max_retries}): {repr(e)}")
                    time.sleep(3)
                else:
                    raise
