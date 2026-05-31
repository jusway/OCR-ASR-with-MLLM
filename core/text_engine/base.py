"""文本模型抽象基类。"""
from abc import ABC, abstractmethod


class BaseTextModel(ABC):
    def __init__(self, model_name: str, temperature: float, top_p: float):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def generate_stream(self, system_prompt: str, prompt: str):
        """流式生成"""
        pass
