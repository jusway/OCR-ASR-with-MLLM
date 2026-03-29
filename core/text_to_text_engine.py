import os
import json
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from google import genai
from google.genai import types


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
    def __init__(self, model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.7, top_p: float = 0.95):
        super().__init__(model_name, temperature, top_p)
        self.client = genai.Client(http_options={'timeout': None})

    def generate(self, system_prompt: str, prompt: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text

    def generate_stream(self, system_prompt: str, prompt: str):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


# ==========================================
# MiMo 实现类 (OpenAI 兼容接口)
# ==========================================
class MiMoText(BaseTextModel):
    def __init__(self, model_name: str = "mimo-v2-pro", temperature: float = 1.0, top_p: float = 0.95,
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
# 主程序
# ==========================================
if __name__ == "__main__":
    system_prompt = (
        "你是一位资深的佛学逐字录音稿整理的比丘师父。"
        "有一个讲法长音频，切分成多个短音频，拼接汇总成了总逐字稿。"
        "你的任务是将得到的总逐字稿整理成书面文稿。"
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
        "讲法者提到的原典文，务必参照【原典参考】进行校对。"
        "引用原典时候使用『』符号框住。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        "不丢失任何总逐字稿的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
    )

    prompt_path = r"../摩诃止观-久仁法师/摩诃止观013/013原文.txt"
    transcript_path = "../摩诃止观-久仁法师/摩诃止观013/摩诃止观013_逐字稿.md"

    # --- 选择模型实例 ---
    # 使用 Gemini:
    text_bot = GeminiText(model_name="gemini-3.1-pro-preview", temperature=0.3)

    # 使用 MiMo-V2-Pro:
    # text_bot = MiMoText(model_name="mimo-v2-pro", temperature=0.3)

    # 读取文件
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            original_text = f.read()
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        exit()

    user_prompt = f"【原典参考】\n{original_text}\n\n【总逐字稿】\n{transcript_text}"

    output_filename = Path(transcript_path).parent / f"{Path(transcript_path).stem.replace('_逐字稿', '')}_书面稿.txt"

    print(f"正在调用 {text_bot.model_name} 处理，请耐心等待...")

    # 示例：使用流式输出并在控制台打印，同时收集全文
    full_text = ""
    for chunk in text_bot.generate_stream(system_prompt, user_prompt):
        print(chunk, end="", flush=True)
        if chunk is not None:
            full_text += chunk

    # 写入文件
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n\n写入完成，已保存至: {output_filename}")