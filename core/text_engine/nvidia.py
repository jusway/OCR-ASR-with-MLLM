"""NVIDIA (英伟达) 实现类。"""
import os
import time
from openai import OpenAI
from core.utils import Color
from .base import BaseTextModel


class NvidiaText(BaseTextModel):
    """使用英伟达 (NVIDIA) API 进行文本处理，支持思考过程。"""
    def __init__(self, model_name: str = "minimaxai/minimax-m2.7", temperature: float = 1.0, top_p: float = 1.0,
                 enable_thinking: bool = False, thinking_kwargs: dict = None):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.enable_thinking = enable_thinking
        self.thinking_kwargs = thinking_kwargs or ({"thinking": True} if enable_thinking else {})
        if not self.api_key:
            raise ValueError("未找到 NVIDIA_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=600.0,
        )

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.thinking_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": self.thinking_kwargs}
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
                    _err_msg = repr(e)
                    _seen = {id(e)}
                    _cause = e.__cause__
                    while _cause and id(_cause) not in _seen:
                        _seen.add(id(_cause))
                        _err_msg += f"\n  └─ {repr(_cause)}"
                        _cause = _cause.__cause__
                    print(f"{Color.RED}⚠️ {self.model_name} 调用失败 (尝试 {attempt+1}/5): {_err_msg}")
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
                    _err_msg = repr(e)
                    _seen = {id(e)}
                    _cause = e.__cause__
                    while _cause and id(_cause) not in _seen:
                        _seen.add(id(_cause))
                        _err_msg += f"\n  └─ {repr(_cause)}"
                        _cause = _cause.__cause__
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {_err_msg}")
                    time.sleep(3)
                else:
                    raise
