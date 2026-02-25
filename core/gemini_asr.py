import os
import time
import uuid
import subprocess
from google import genai
from google.genai import types
from pathlib import Path


class GeminiASR:
    def __init__(self, model_name: str, temperature: float = 0.1, top_p: float = 0.95):
        self.client = genai.Client(http_options={'timeout': None})
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str):
        audio_file = self.client.files.upload(file=audio_file_path)

        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = self.client.files.get(name=audio_file.name)

        if audio_file.state.name == "FAILED":
            raise RuntimeError(f"云端音频文件处理失败: {audio_file.name}")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=65536,
        )

        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=[audio_file, prompt],
            config=config
        )

        full_text = ""
        for chunk in response_stream:
            full_text += chunk.text

        print()

        self.client.files.delete(name=audio_file.name)
        return full_text


if __name__ == "__main__":
    system_prompt = (
        # 角色
        "你是顶尖的佛法音频转录专家。"
        # 任务
        "将用户提供的音频准确转录为Markdown格式逐字稿。"
        # 输出规范
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "使用简体字输出。"
        "原典引用的时候使用繁体字。"
    )

    prompt_path = r"../摩诃止观-久仁法师-各音频原典提示词/001.txt"  # 记载原文内容
    audio_path = "../摩诃止观-久仁法师/摩诃止观001.mp3"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    prompt = f"""
    【原典参考】（由pdf复制而来）
    {prompt_text}
    """

    asr = GeminiASR(model_name="gemini-3.1-pro-preview", temperature=0.1, top_p=0.95)

    print("正在提交给模型处理，采用流式输出，请耐心等待...")
    full_transcription = asr.recognize(system_prompt, prompt, audio_path)
    print("处理完成！")

    output_filename = f"{Path(audio_path).stem}_逐字稿.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_transcription)