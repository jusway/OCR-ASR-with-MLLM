"""PackyCode 中转站 API 实现类。OpenAI 兼容格式。"""
import os
import time
from openai import OpenAI
from core.utils import Color
from .base import BaseTextModel


class PackyCodeText(BaseTextModel):
    """
    使用 PackyCode 中转站 API 进行文本处理。
    支持分组信息（group）注入，使用代理模型名称。

    配置：
      环境变量 PackyCode_API_KEY 作为密钥。
      端点为 https://www.packyapi.com
    """

    def __init__(self, model_name: str = "deepseek-v4-pro",
                 temperature: float = 1.0, top_p: float = 1.0,
                 enable_thinking: bool = False, reasoning_effort: str = "high",
                 thinking_effort: str = None,
                 group: str = None, extra_body: dict = None):
        super().__init__(model_name, temperature, top_p)
        self.api_key = self._resolve_api_key(group)
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.thinking_effort = thinking_effort
        self.group = group
        self.extra_body_inject = extra_body or {}
        self._last_reasoning_content = None
        if not self.api_key:
            raise ValueError("未找到 PackyCode_API_KEY，请设置环境变量。")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://www.packyapi.com/v1",
            timeout=600.0,
        )

    @staticmethod
    def _resolve_api_key(group: str = None) -> str:
        """按分组查找对应密钥，未找到则回退到默认 PackyCode_API_KEY。"""
        if group:
            group_key = f"PackyCode_API_KEY_{group.upper().replace('-', '_')}"
            key = os.getenv(group_key)
            if key:
                return key
        return os.getenv("PackyCode_API_KEY")

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        # 构建 extra_body
        body = {}
        if self.group:
            body["group"] = self.group
        body.update(self.extra_body_inject)
        if self.thinking_effort is not None:
            # 自适应思考（Claude Opus 4.7+）
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            body["thinking"] = {"type": "adaptive", "effort": self.thinking_effort}
        elif self.enable_thinking:
            body["reasoning_effort"] = self.reasoning_effort
            body["thinking"] = {"type": "enabled"}
        else:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            body["thinking"] = {"type": "disabled"}
        if body:
            kwargs["extra_body"] = body
        return kwargs

    def _build_messages(self, system_prompt: str, prompt: str) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
                    print(f"\n{Color.RED}⚠️ {self.model_name} (PackyCode) 流式调用失败 (尝试 {attempt+1}/5): {_err_msg}")
                    time.sleep(3)
                else:
                    raise


class CodexPackyCodeText(PackyCodeText):
    """codex 分组 — GPT 系列模型，不传 thinking 参数。"""
    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        body = {}
        if self.group:
            body["group"] = self.group
        body.update(self.extra_body_inject)
        if body:
            kwargs["extra_body"] = body
        return kwargs


class CCPackyCodeText(PackyCodeText):
    """cc 分组 — Claude nothink 模式，固定禁用思考。"""
    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        body = {"thinking": {"type": "disabled"}}
        if self.group:
            body["group"] = self.group
        body.update(self.extra_body_inject)
        kwargs["extra_body"] = body
        return kwargs


class AWSQPackyCodeText(PackyCodeText):
    """aws-q 分组 — Claude adaptive / nothink 模式。"""
    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        body = {}
        if self.thinking_effort is not None:
            body["thinking"] = {"type": "adaptive", "effort": self.thinking_effort}
        elif self.enable_thinking:
            body["thinking"] = {"type": "enabled"}
        else:
            body["thinking"] = {"type": "disabled"}
        if self.group:
            body["group"] = self.group
        body.update(self.extra_body_inject)
        kwargs["extra_body"] = body
        return kwargs


class CCSalePackyCodeText(PackyCodeText):
    """cc-sale 分组 — nothink + adaptive 混合，行为同基类。"""
    pass
