"""Kimi (Moonshot) 实现类。"""
import os
import sys
import time
import threading
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
    - 内置流式超时检测：思考模式 3 分钟/普通模式 1 分钟内无新 chunk 则退出。
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
            base_url="https://api.moonshot.cn/v1",
            timeout=600.0,
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

    def _timeout_iter(self, stream, chunk_timeout: float):
        """迭代流式响应，若 chunk_timeout 秒内无新 chunk 则抛 TimeoutError。"""
        ev = threading.Event()
        result = {"value": None, "error": None, "done": False}

        def fetch():
            try:
                result["value"] = next(stream)
            except StopIteration:
                result["done"] = True
            except Exception as e:
                result["error"] = e
            ev.set()

        t = threading.Thread(target=fetch, daemon=True)
        t.start()
        if not ev.wait(chunk_timeout):
            raise TimeoutError(f"Kimi {chunk_timeout}s 无响应")
        if result["error"]:
            raise result["error"]
        if result["done"]:
            raise StopIteration
        return result["value"]

    def generate(self, system_prompt: str, prompt: str) -> str:
        messages = self._build_messages(system_prompt, prompt)
        chunk_timeout = 180 if self.enable_thinking else 60
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    **self._get_kwargs(messages, stream=True)
                )
                content_parts = []
                reasoning_parts = []
                try:
                    while True:
                        chunk = self._timeout_iter(response, chunk_timeout)
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if not delta:
                            continue
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            reasoning_parts.append(delta.reasoning_content)
                            continue
                        if delta.content:
                            content_parts.append(delta.content)
                except StopIteration:
                    pass
                self._last_reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
                return "".join(content_parts)
            except TimeoutError:
                print(f"{Color.RED}❌ {self.model_name} {chunk_timeout}s 内无响应，终止运行{Color.END}")
                sys.exit(1)
            except Exception as e:
                if attempt < 4:
                    print(f"{Color.RED}⚠️ {self.model_name} 调用失败 (尝试 {attempt+1}/5): {repr(e)}{Color.END}")
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
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {repr(e)}")
                    time.sleep(3)
                else:
                    raise
