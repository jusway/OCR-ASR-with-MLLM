"""NVIDIA (英伟达) 实现类。"""
import os
import time
from openai import OpenAI
from core.utils import Color
from .base import BaseTextModel


class NvidiaText(BaseTextModel):
    """使用英伟达 (NVIDIA) API 进行文本处理，支持思考过程。"""
    def __init__(self, model_name: str = "minimaxai/minimax-m2.7", temperature: float = 1.0, top_p: float = 1.0,
                 enable_thinking: bool = False):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.enable_thinking = enable_thinking
        if not self.api_key:
            raise ValueError("未找到 NVIDIA_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
        else:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs

    def _build_messages(self, system_prompt: str, prompt: str) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, system_prompt: str, prompt: str) -> str:
        messages = self._build_messages(system_prompt, prompt)
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(**self._get_kwargs(messages, stream=False))
                if response and response.choices:
                    msg = response.choices[0].message
                    self._last_reasoning_content = msg.reasoning_content if hasattr(msg, 'reasoning_content') else None
                    return msg.content or ""
            except Exception as e:
                if attempt < 4:
                    print(f"{Color.RED}⚠️ {self.model_name} 调用失败 (尝试 {attempt+1}/5): {e}")
                    time.sleep(3)
                else:
                    raise

    def generate_stream(self, system_prompt: str, prompt: str):
        messages = self._build_messages(system_prompt, prompt)
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(**self._get_kwargs(messages, stream=True))
                reasoning_parts = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_parts.append(delta.reasoning_content)
                        continue
                    if delta.content:
                        yield delta.content
                self._last_reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
                return
            except Exception as e:
                if attempt < 4:
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {e}")
                    time.sleep(3)
                else:
                    raise
