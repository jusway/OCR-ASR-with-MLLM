"""Kimi (Moonshot) 实现类。"""
import os
import time
from openai import OpenAI
from core.utils import Color
from .base import BaseTextModel


class KimiText(BaseTextModel):
    """
    使用 Kimi (Moonshot) API 进行文本处理。
    支持 kimi-k2.6 / kimi-k2.5（可开关思考）和 kimi-k2-thinking（强制思考）。

    注意：
    - K2.6/K2.5 的 temperature/top_p 由 API 固定，不可传参，传了会报错。
    - 思考模式通过 extra_body 控制，无 reasoning_effort 概念。
    """
    def __init__(self, model_name: str = "kimi-k2.6",
                 enable_thinking: bool = True):
        super().__init__(model_name, temperature=1.0, top_p=0.95)
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.enable_thinking = enable_thinking
        self._last_reasoning_content = None
        if not self.api_key:
            raise ValueError("未找到 MOONSHOT_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.cn/v1"
        )

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": 32768,
        }
        if self.model_name in ("kimi-k2.6", "kimi-k2.5"):
            if not self.enable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["temperature"] = 1.0
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
                response = self.client.chat.completions.create(
                    **self._get_kwargs(messages, stream=False)
                )
                if response and response.choices:
                    msg = response.choices[0].message
                    self._last_reasoning_content = getattr(msg, "reasoning_content", None)
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
                response = self.client.chat.completions.create(
                    **self._get_kwargs(messages, stream=True)
                )
                reasoning_parts = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    if hasattr(delta, "reasoning_content"):
                        rc = getattr(delta, "reasoning_content")
                        if rc:
                            reasoning_parts.append(rc)
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
