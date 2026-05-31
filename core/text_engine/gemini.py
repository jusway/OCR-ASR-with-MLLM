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
                    _err_msg = repr(e)
                    _seen = {id(e)}
                    _cause = e.__cause__
                    while _cause and id(_cause) not in _seen:
                        _seen.add(id(_cause))
                        _err_msg += f"\n  └─ {repr(_cause)}"
                        _cause = _cause.__cause__
                    print(f"\n{Color.RED}⚠️ Gemini API 流式调用失败 (尝试 {attempt + 1}/{max_retries}): {_err_msg}")
                    time.sleep(3)
                else:
                    raise