import os
from pathlib import Path
from google import genai
from google.genai import types


class GeminiText:
    def __init__(self, model_name: str = "gemini-3-pro-preview", temperature: float = 0.7, top_p: float = 0.95):
        self.client = genai.Client(http_options={'timeout': None})
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

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


if __name__ == "__main__":
    system_prompt = (
        # 角色
        "你是一位资深的佛学逐字录音稿整理义工。"
        # 任务
        "有一个讲法长音频，切分成多个短音频，分配给多个义工去做了语音转录，拼接汇总成了总逐字稿。"
        "你的任务是将得到的总逐字稿整理成书面文稿。"
        # 风格
        "风格要大白话、口语化，不怕字数多，但尽量使再笨的人也能看懂。"
        # 参考原典
        "讲法者提到的原典文，务必参照【原典参考】进行校对。"
        # 格式规范
        "引用原典时候使用『』符号框住。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        # 信息规范
        "义工的逐字稿可能会听写有误，遇到可能是谬误的字词，酌情改正。"
        "不丢失任何总逐字稿的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
    )
    prompt_path = r"摩诃止观-久仁法师/摩诃止观005/005原文.txt"
    transcript_path = "摩诃止观-久仁法师/摩诃止观005/摩诃止观005_逐字稿.md"
    model_name="gemini-3-pro-preview"



    with open(prompt_path, "r", encoding="utf-8") as f:
        original_text = f.read()
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    user_prompt = f"""
【原典参考】
{original_text}

【总逐字稿】
{transcript_text}
"""

    text_bot = GeminiText(model_name=model_name, temperature=0.3)

    output_filename = Path(transcript_path).parent / f"{Path(transcript_path).stem.replace('_逐字稿', '')}_书面稿.md"

    print("正在提交给模型处理，因为采用非流式等待，控制台暂无输出，请耐心等待...")
    full_text = text_bot.generate(system_prompt, user_prompt)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"写入完成，已保存至: {output_filename}")