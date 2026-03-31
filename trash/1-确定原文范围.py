
import os
import uuid
from pathlib import Path
from pydub import AudioSegment
from core.asr_engine import GeminiASR

class AudioTailTextLocator:
    def __init__(self, model_name, tail_minutes=20, temperature=0.1, top_p=0.95):
        self.asr = GeminiASR(temperature=temperature, top_p=top_p, model_name=model_name)
        self.tail_minutes = tail_minutes

    def _extract_tail(self, audio_path: str) -> str:
        audio = AudioSegment.from_file(audio_path)
        tail_ms = self.tail_minutes * 60 * 1000

        if len(audio) < tail_ms:
            chunk = audio
        else:
            chunk = audio[-tail_ms:]

        out_path = f"temp_tail_{uuid.uuid4().hex}.mp3"
        chunk.export(out_path, format="mp3")

        return out_path

    def process(self, audio_path: str, text_path: str) -> str:
        temp_audio_path = self._extract_tail(audio_path)

        with open(text_path, "r", encoding="utf-8") as f:
            reference_text = f.read()

        system_prompt = (
            # 角色
            "你是一位资深的佛法音频校对助手。"
            # 任务
            "听取提供的音频，并阅读提供的【原文模糊范围】。"
            "请精准定位并输出这段音频刚好在讲解【原文模糊范围】中的哪一部分。"
            # 信息规范
            "只输出这段原文文本。"
            "不要说多余的话。"
        )

        prompt = f"【原文模糊范围】\n{reference_text}"

        print(f"正在截取音频最后 {self.tail_minutes} 分钟并提交给模型定位...")
        location_result = self.asr.recognize(system_prompt, prompt, temp_audio_path)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return location_result

if __name__ == "__main__":
    audio_path = "../摩诃止观-久仁法师/摩诃止观019/摩诃止观019.mp3"
    text_path = "../摩诃止观-久仁法师/摩诃止观019/019原文模糊范围.txt"
    target_minutes = 600
    model_name = "gemini-3-pro-preview"

    locator = AudioTailTextLocator(model_name=model_name, tail_minutes=target_minutes)
    final_location = locator.process(audio_path, text_path)

    audio_p = Path(audio_path)
    output_filename = audio_p.parent / f"{audio_p.stem}_原文定位.md"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_location)

    print(f"定位完成！结果已保存至：{output_filename}")