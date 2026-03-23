import os
from pathlib import Path
from pydub import AudioSegment

from core.batch_audio_transcriber import BatchAudioTranscriber
from core.gemini_text import GeminiText


class AudioProcessingPipeline:
    def __init__(self, model_name, chunk_minutes=20, overlap_minutes=10, max_workers=8):
        self.model_name = model_name
        self.chunk_minutes = chunk_minutes
        self.overlap_minutes = overlap_minutes
        self.max_workers = max_workers

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, audio_path, prompt_path, asr_sys_prompt, text_sys_prompt):
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        asr_prompt = f"\n    【原典参考】\n    {prompt_text}\n    "

        transcriber = BatchAudioTranscriber(
            model_name=self.model_name,
            chunk_minutes=self.chunk_minutes,
            overlap_seconds=self.overlap_minutes * 60,
            temperature=0.1,
            top_p=0.95,
            max_workers=self.max_workers
        )

        print(f"[1/3] 开始进行音频切分与逐字稿生成...")
        transcript_text = transcriber.process_full_audio(audio_path, asr_sys_prompt, asr_prompt)

        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"逐字稿已保存至: {transcript_filename}")

        text_bot = GeminiText(model_name=self.model_name, temperature=0.3)
        user_prompt = f"\n【原典参考】\n{prompt_text}\n\n【总逐字稿】\n{transcript_text}\n"

        print("\n[2/3] 正在将逐字稿整理为书面稿，请耐心等待...")
        written_text = text_bot.generate(text_sys_prompt, user_prompt)

        print("\n[3/3] API 处理全部完成！请补充元数据信息以生成最终文档：")
        year = input("请输入音频的创建时间（年份）: ")
        author = input("请输入音频的作者: ")
        metadata_original_text = input("请输入音频的原文: ")

        duration = self._get_audio_duration(audio_path)
        metadata = (
            f"> 标题：{base_name}\n"
            f"> 时间：{year}\n"
            f"> 时长：{duration}\n"
            f"> 作者：{author}\n"
            f"> 原文：{metadata_original_text}\n"
            "\n"
            "---\n\n"
        )

        final_content = metadata + written_text

        with open(final_output_filename, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"\n全部完成！带元数据的书面稿已成功写入：{final_output_filename}")


if __name__ == "__main__":
    asr_system_prompt = (
        # 角色
        "你是一位资深的佛法音频转录工作的出家比丘大师。"
        # 任务
        "将用户提供的音频准确转录为逐字稿。"
        # 输出规范
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "引用原典的时候使用『』符号。"
        "只输出转录内容，不说多余的话。"
    )

    text_system_prompt = (
        # 角色
        "你是一位资深的佛学逐字录音稿整理的比丘师父。"
        # 任务
        "有一个讲法长音频，切分成多个短音频，相邻的短音频互有重叠，分配给了多个义工去做了语音转录，拼接汇总成了总逐字稿。"
        "你的任务是将得到的总逐字稿整理成书面文稿。"
        # 风格
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
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

    prompt_path = r"摩诃止观-久仁法师/摩诃止观018/摩诃止观018_原文定位.md"
    audio_path = "摩诃止观-久仁法师/摩诃止观018/摩诃止观018.mp3"
    model_name = "gemini-3.1-pro-preview"

    pipeline = AudioProcessingPipeline(
        model_name=model_name,
        chunk_minutes=20,
        overlap_minutes=10,
        max_workers=8
    )

    pipeline.run(audio_path, prompt_path, asr_system_prompt, text_system_prompt)