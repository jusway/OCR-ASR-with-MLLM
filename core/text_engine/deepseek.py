"""DeepSeek 官方 API 实现类。"""
import os
import time
from openai import OpenAI
from core.utils import Color
from .base import BaseTextModel


class DeepSeekText(BaseTextModel):
    """
    使用 DeepSeek 官方 API 进行文本处理。
    支持 reasoning_effort 参数控制思考深度。
    """
    def __init__(self, model_name: str = "deepseek-v4-pro", temperature: float = 1.0, top_p: float = 0.95,
                 enable_thinking: bool = False, reasoning_effort: str = "high"):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self._last_reasoning_content = None
        if not self.api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            timeout=600.0,
        )

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream
        }
        if self.enable_thinking:
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
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
                    print(f"{Color.RED}⚠️ {self.model_name} 调用失败 (尝试 {attempt+1}/5): {repr(e)}")
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
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {repr(e)}")
                    time.sleep(3)
                else:
                    raise
