import os
import json
import requests
import warnings
import time
from abc import ABC, abstractmethod
from pathlib import Path
from google import genai
from google.genai import types
from google.genai import errors
from openai import OpenAI
from core.utils import Color


# ==========================================
# 抽象基类
# ==========================================
class BaseTextModel(ABC):
    def __init__(self, model_name: str, temperature: float, top_p: float):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def generate(self, system_prompt: str, prompt: str) -> str:
        """非流式生成"""
        pass

    @abstractmethod
    def generate_stream(self, system_prompt: str, prompt: str):
        """流式生成"""
        pass


# ==========================================
# Gemini 实现类
# ==========================================
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
                    print(f"{Color.RED}⚠️ Gemini API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"{Color.RED}等待 3 秒后重试...\033[0m")
                    time.sleep(3)
                else:
                    print(f"{Color.RED}❌ Gemini API 调用失败，已达到最大重试次数 ({max_retries}次)。\033[0m")
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
                break  # 如果成功流式输出完毕，跳出重试循环
            except errors.APIError as e:
                if attempt < max_retries - 1:
                    print(f"\n{Color.RED}⚠️ Gemini API 流式调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"{Color.RED}等待 3 秒后重试...\033[0m")
                    time.sleep(3)
                else:
                    print(f"\n{Color.RED}❌ Gemini API 流式调用失败，已达到最大重试次数 ({max_retries}次)。\033[0m")
                    raise


# ==========================================
# Doubao (火山引擎) 实现类
# ==========================================
class DoubaoText(BaseTextModel):
    """
    使用火山引擎 (豆包) 模型进行文本处理。
    """
    def __init__(self, model_name: str = "doubao-seed-2-0-pro-260215", temperature: float = 0.3, top_p: float = 0.95):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 ARK_API_KEY，请设置环境变量。")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

    def generate(self, system_prompt: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p
        )
        return response.choices[0].message.content

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


# ==========================================
# MiMo 实现类 (OpenAI 兼容接口)
# ==========================================
class MiMoText(BaseTextModel):
    """
    使用 MiMo 模型进行文本处理。
    """
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
                {"role": "developer", "content": system_prompt},  # MiMo文档推荐使用developer角色
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
            "thinking": {"type": "disabled"}  # 默认关闭思维链以获取纯文本结果，如需开启可设为enabled
        }

    def generate(self, system_prompt: str, prompt: str) -> str:
        payload = self._build_payload(system_prompt, prompt, stream=False)
        response = requests.post(self.base_url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']

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


# ==========================================
# 英伟达 (NVIDIA) 实现类
# ==========================================
class NvidiaText(BaseTextModel):
    """
    使用英伟达 (NVIDIA) API 进行文本处理。
    支持带有思考过程 (Reasoning) 的模型。
    """
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
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
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
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    # 思考模式：跳过 reasoning_content，只产出 content
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        continue
                    if delta.content:
                        yield delta.content
                return
            except Exception as e:
                if attempt < 4:
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {e}")
                    time.sleep(3)
                else:
                    raise


# ==========================================
# DeepSeek 官方 API 实现类
# ==========================================
class DeepSeekText(BaseTextModel):
    """
    使用 DeepSeek 官方 API 进行文本处理。
    支持 reasoning_effort 参数控制思考深度。
    """
    def __init__(self, model_name: str = "deepseek-v4-pro", temperature: float = 1.0, top_p: float = 0.95,
                 enable_thinking: bool = False, reasoning_effort: str = "high", max_tokens: int = 0):
        super().__init__(model_name, temperature, top_p)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self._last_reasoning_content = None
        if not self.api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY，请设置环境变量。")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def _get_kwargs(self, messages: list, stream: bool) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream
        }
        if self.max_tokens > 0:
            kwargs["max_tokens"] = self.max_tokens
        if self.enable_thinking:
            # 思考模式下 temperature / top_p 不生效，不传更干净
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
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
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    # 思考模式：跳过 reasoning_content，只产出 content
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        continue
                    if delta.content:
                        yield delta.content
                return
            except Exception as e:
                if attempt < 4:
                    print(f"\n{Color.RED}⚠️ {self.model_name} 流式调用失败 (尝试 {attempt+1}/5): {e}")
                    time.sleep(3)
                else:
                    raise
